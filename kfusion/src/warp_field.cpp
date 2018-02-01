#include <dual_quaternion.hpp>
#include <knn_point_cloud.hpp>
#include <kfusion/types.hpp>
#include <nanoflann.hpp>
#include "kfusion/warp_field.hpp"
#include "internal.hpp"
#include "precomp.hpp"
#include <opencv2/core/affine.hpp>
#include <kfusion/optimisation.hpp>
#include <cmath>
#include <opencv2/core/core.hpp>

using namespace kfusion;
std::vector<utils::DualQuaternion<double>> neighbours; //THIS SHOULD BE SOMEWHERE ELSE BUT TOO SLOW TO REINITIALISE
utils::PointCloud cloud;
nanoflann::KNNResultSet<float> *resultSet_;
std::vector<float> out_dist_sqr_;
std::vector<size_t> ret_index_;

WarpField::WarpField()
{
    nodes_ = new std::vector<deformation_node>();
    index_ = new kd_tree_t(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    ret_index_ = std::vector<size_t>(KNN_NEIGHBOURS);
    out_dist_sqr_ = std::vector<float>(KNN_NEIGHBOURS);
    resultSet_ = new nanoflann::KNNResultSet<float>(KNN_NEIGHBOURS);
    resultSet_->init(&ret_index_[0], &out_dist_sqr_[0]);
    neighbours = std::vector<utils::DualQuaternion<double>>(KNN_NEIGHBOURS);
    warp_to_live_ = cv::Affine3f();

}

WarpField::~WarpField()
{
    delete nodes_;
    delete resultSet_;
    delete index_;
}

/**
 *
 * @param first_frame
 * @param normals
 */
void WarpField::init(const cv::Mat& first_frame)
{
    nodes_->resize(first_frame.cols * first_frame.rows);
    auto voxel_size = kfusion::KinFuParams::default_params_dynamicfusion().volume_size[0] /
                      kfusion::KinFuParams::default_params_dynamicfusion().volume_dims[0];

//    FIXME:: this is a test, remove later
    voxel_size = 1;
    int step = 10;
    for(size_t i = 0; i < first_frame.rows; i+=step)
        for(size_t j = 0; j < first_frame.cols; j+=step)
        {
            auto point = first_frame.at<Point>(i,j);
            if(!std::isnan(point.x))
            {
                
                auto t = utils::Quaternion<double>(0,point.x,point.y,point.z);
                nodes_->at(i*first_frame.cols+j).transform = utils::DualQuaternion<double>(t, utils::Quaternion<double>());
                nodes_->at(i*first_frame.cols+j).vertex = Vec3f(point.x,point.y,point.z);
                nodes_->at(i*first_frame.cols+j).weight = 3 * voxel_size;
            }
        }
    buildKDTree();
}

/**
 *
 * @param first_frame
 * @param normals
 */
void WarpField::init(const std::vector<Vec3f>& first_frame)
{
    nodes_->resize(first_frame.size());
    auto voxel_size = kfusion::KinFuParams::default_params_dynamicfusion().volume_size[0] /
                      kfusion::KinFuParams::default_params_dynamicfusion().volume_dims[0];

//    FIXME: this is a test, remove
    voxel_size = 1;
    for (size_t i = 0; i < first_frame.size(); i++)
    {
        auto point = first_frame[i];
        if (!std::isnan(point[0]))
        {
            //YuYang
            // utils::Quaternion<float> t(0.f, point[0], point[1], point[2]);
            // nodes_->at(i).transform = utils::DualQuaternion<float>(t,utils::Quaternion<float>());
            utils::Quaternion<double> t(0.f, point[0], point[1], point[2]);
            nodes_->at(i).transform = utils::DualQuaternion<double>(t,utils::Quaternion<double>());

            nodes_->at(i).vertex = point;
            nodes_->at(i).weight = 3 * voxel_size;
        }
    }
    buildKDTree();
}

/**
 * \brief
 * \param frame
 * \param normals
 * \param pose
 * \param tsdfVolume
 * \param edges
 */
void WarpField::energy(const cuda::Cloud &frame,
                       const cuda::Normals &normals,
                       const Affine3f &pose,
                       const cuda::TsdfVolume &tsdfVolume,
                       const std::vector<std::pair<utils::DualQuaternion<double>, utils::DualQuaternion<double>>> &edges
)
{
    assert(normals.cols()==frame.cols());
    assert(normals.rows()==frame.rows());
}

/**
 *
 * @param canonical_vertices
 * @param canonical_normals
 * @param live_vertices
 * @param live_normals
 * @return
 */

// [Minhui 2018/1/28]modified
void WarpField::energy_data(const std::vector<Vec3d> &canonical_vertices,
                            const std::vector<Vec3d> &canonical_normals,
                            const std::vector<Vec3d> &live_vertices,
                            const std::vector<Vec3d> &live_normals,
                            const Intr& intr
)
{

//    assert((canonical_normals.size() == canonical_vertices.size()) == (live_normals.size() == live_vertices.size()));
    ceres::Problem problem;
    float weights[KNN_NEIGHBOURS];
    unsigned long indices[KNN_NEIGHBOURS];

    WarpProblem warpProblem(this);
    live_vertices_ = live_vertices; // [Minhui] using the whole point cloud as parameter of cost funciton would cause RAM rising rapidly

    for(int i = 0; i < live_vertices.size(); i++)
    {
        if(std::isnan(canonical_vertices[i][0]) || std::isnan(canonical_normals[i][0])) {
            continue;
        }
        getWeightsAndUpdateKNN(canonical_vertices[i], weights); // [Minhui 2018/1/28]would update the weights and ret_index_(might be index of KNN in nodes_)

        //        FIXME: could just pass ret_index
        for(int j = 0; j < KNN_NEIGHBOURS; j++) {
            indices[j] = ret_index_[j];
        }

        cv::Vec3d canonical_point = canonical_vertices[i];
        cv::Vec3d canonical_point_n = canonical_normals[i];
        
        /*    ************test************   */
        // [Step 2] Calculte the dqb warp function of the canonical point
        // Reference to warpField_->DQB_r
        kfusion::utils::Quaternion<double> translation_sum(0.0, 0.0, 0.0, 0.0);
        kfusion::utils::Quaternion<double> rotation_sum(0.0, 0.0, 0.0, 0.0);

        float weights_sum = 0;
        for(int m = 0; m < KNN_NEIGHBOURS; m++) {
            weights_sum += weights[m];
        }
        for(int m = 0; m < KNN_NEIGHBOURS; m++) {
            weights[m] = weights[m] / weights_sum;
        }

        for(int m = 0; m < KNN_NEIGHBOURS; m++) {
            kfusion::utils::DualQuaternion<double> temp;
            kfusion::utils::Quaternion<double> rotation(0.0f, 0.0f, 0.0f, 0.0f);
            kfusion::utils::Quaternion<double> translation(0.0f, 0.0f, 0.0f, 0.0f);
            //printf("Debug\n");
            nodes_->at(indices[m]).transform.rotation_.w_ = 1;
            temp.rotation_.x_ = nodes_->at(indices[m]).transform.rotation_.x_;
            temp.rotation_.y_ = nodes_->at(indices[m]).transform.rotation_.y_;
            temp.rotation_.z_ = nodes_->at(indices[m]).transform.rotation_.z_;
            temp.translation_.w_ = nodes_->at(indices[m]).transform.translation_.w_;
            temp.translation_.x_ = nodes_->at(indices[m]).transform.translation_.x_;
            temp.translation_.y_ = nodes_->at(indices[m]).transform.translation_.y_;
            temp.translation_.z_ = nodes_->at(indices[m]).transform.translation_.z_;

            rotation = temp.getRotation();
            translation = temp.getTranslation();

            rotation_sum += weights[m] * rotation;
            translation_sum += weights[m] * translation;

        }
        // kfusion::utils::DualQuaternion<double> dqb;
        // dqb.from_twist(rotation_sum.x_, rotation_sum.y_, rotation_sum.z_,
                            //translation_sum.x_, translation_sum.y_, translation_sum.z_);
        double sinr = 2.0 * (rotation_sum.w_ * rotation_sum.x_ +
                         rotation_sum.y_ * rotation_sum.z_);
        double cosr = 1.0 - 2.0 * (pow(rotation_sum.x_, 2) + pow(rotation_sum.y_, 2));
        double siny = 2.0 * (rotation_sum.w_ * rotation_sum.z_ +
                            rotation_sum.x_ * rotation_sum.y_);
        double cosy = 1.0 - 2.0 * (rotation_sum.y_ * rotation_sum.y_ +
                                rotation_sum.z_ * rotation_sum.z_);
        double sinp = 2.0 * (rotation_sum.w_ * rotation_sum.y_ - rotation_sum.z_ * rotation_sum.x_);

        cv::Vec3d i_rot;
        if (fabs(sinp) >= 1)
            cv::Vec3d i_rot(atan2(sinr, cosr), copysign(M_PI / 2, sinp), atan2(siny, cosy));
        else
            cv::Vec3d i_rot(atan2(sinr, cosr), asin(sinp), atan2(siny, cosy));
        cv::Vec3d i_trans(translation_sum.x_, translation_sum.y_, translation_sum.z_);
        cv::Affine3d i_warp(i_rot, i_trans);
        
        canonical_point =  i_warp * canonical_point;
        canonical_point_n = i_warp * canonical_point_n;

        // dqb.transform(canonical_point);
        // dqb.transform(canonical_point_n);

        // [Step 2] project the 3D conanical point to live-frame image domain (2D-coordinate)
        cv::Vec2d project_point = project(canonical_vertices[i][0], canonical_vertices[i][1], canonical_vertices[i][2], intr); //Is point.z needs to be in homogeneous coordinate? (point.z==1)

        double project_u = project_point[0];
        double project_v = project_point[1];
        double depth = 0.0f;

        depth = live_vertices[project_u * 640 + project_v][2];
        if(std::isnan(depth) || project_u >= 480 || project_u < 0 || project_v >= 640 || project_v < 0 ) {
            continue;
        }
        /*    ************test************   */

        ceres::CostFunction* cost_function = DynamicFusionDataEnergy::Create(live_vertices[i],
                                                                             live_normals[i],
                                                                             canonical_vertices[i],
                                                                             canonical_normals[i],
                                                                             this,
                                                                             weights,
                                                                             indices,
                                                                             intr,
                                                                             i);
        ceres::HuberLoss *loss_function = new ceres::HuberLoss(1.0); 
        //ceres::TukeyLoss *loss_function = new ceres::TukeyLoss(0.01);
        // for(int m = 0; m < KNN_NEIGHBOURS; m++) {
        //     nodes_->at(indices[m]).transform.rotation_.w_ = 1;
        //     nodes_->at(indices[m]).transform.rotation_.x_ = 2;
        //     nodes_->at(indices[m]).transform.rotation_.y_ = 3;
        //     nodes_->at(indices[m]).transform.rotation_.z_ = 4;
        //     nodes_->at(indices[m]).transform.translation_.w_ = 5;
        //     nodes_->at(indices[m]).transform.translation_.x_ = 6;
        //     nodes_->at(indices[m]).transform.translation_.y_ = 7;
        //     nodes_->at(indices[m]).transform.translation_.z_ = 8;
            // printf("%f %f %f %f\n", p[m][0], p[m][1], p[m][2], p[m][3]);
            // printf("%f %f %f %f\n\n", nodes_->at(indices[m]).transform.rotation_.w_, nodes_->at(indices[m]).transform.rotation_.x_
            //                       , nodes_->at(indices[m]).transform.rotation_.y_, nodes_->at(indices[m]).transform.rotation_.z_);
        // } 
        problem.AddResidualBlock(cost_function, loss_function /* squared loss */, warpProblem.mutable_epsilon(indices));
        //problem.AddResidualBlock(cost_function,  NULL /* squared loss */, warpField_->getNodes()->at(indices[0]).transform);
        //test
        // std::vector<double*> p = warpProblem.mutable_epsilon(indices);
        // for(int m = 0; m < KNN_NEIGHBOURS; m++) {
        //     printf("%f %f %f %f\n", p[m][0], p[m][1], p[m][2], p[m][3]);
        //     printf("%f %f %f %f\n\n", nodes_->at(indices[m]).transform.rotation_.w_, nodes_->at(indices[m]).transform.rotation_.x_
        //                           , nodes_->at(indices[m]).transform.rotation_.y_, nodes_->at(indices[m]).transform.rotation_.z_);
        // }       
    }
    printf("Debug 1\n");
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.num_linear_solver_threads = 12;
    options.num_threads = 12;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    printf("Debug 2\n");
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    // auto params = warpProblem.params();
    // for(int i = 0; i < nodes_->size()*8; i++)
    // {
    //     std::cout<<params[i]<<" ";
    //     if((i+1) % 8 == 0)
    //         std::cout<<std::endl;
    // }

    update_nodes(warpProblem.params());
}

cv::Vec2d WarpField::project(const double &x, const double &y, const double &z, const Intr& intr_) const {
    cv::Vec2d project_point;

    project_point[0] = intr_.fx * (x / z) + intr_.cx;
    project_point[1] = intr_.fy * (y / z) + intr_.cy;

    return project_point;
}

cv::Vec3d WarpField::reproject(const double &u, const double &v, const double &depth, const Intr& intr_) const {
    cv::Vec3d reproject_point;

    reproject_point[0] = depth * (u - intr_.cx) * intr_.fx;
    reproject_point[1] = depth * (v - intr_.cy) * intr_.fy;
    reproject_point[2] = depth;

    return reproject_point;
}


/**
 * \brief
 * \param edges
 */

void WarpField::energy_reg(const std::vector<cv::Vec3f>& surface_points, cv::Affine3f inverse_pose, const std::vector<cv::Matx33f>& warp_rot, const std::vector<cv::Vec3f>& warp_trans){
  float weights[KNN_NEIGHBOURS];
  std::vector<double*> epsilon_transforms;
  float delta = 0.0001;
  float sum = 0;
  WarpProblem warpProblem(this);

  for(int i=0; i<surface_points.size(); ++i){
    getWeightsAndUpdateKNN(surface_points[i], weights);
    //FIXME: This might be KNN of all point, should be KNN of TSDF=0 set.
    //epsilon_transforms = warpProblem.mutable_epsilon(ret_index_);
    float sum_j = 0;
    cv::Affine3f warp(warp_rot[i], warp_trans[i]);
    cv::Affine3f Tic;
    Tic = inverse_pose.concatenate(warp);
    for(int j=0; j<KNN_NEIGHBOURS; ++j){
      auto jpoint = nodes_->at(ret_index_[j]).vertex;
      cv::Vec3f j_rot(epsilon_transforms[j][0],
                      epsilon_transforms[j][1],
                      epsilon_transforms[j][2]);
      cv::Vec3f j_trans(epsilon_transforms[j][3],
                        epsilon_transforms[j][4],
                        epsilon_transforms[j][5]);
      // std::cout << j_rot << std::endl;
      cv::Affine3f j_warp(j_rot, j_trans);
      cv::Affine3f Tjc;
      Tjc = inverse_pose.concatenate(j_warp);
      auto difference = Tic * jpoint - Tjc * jpoint;
      float dist = sqrt(pow(difference[0], 2) +
                        pow(difference[1], 2) +
                        pow(difference[2], 2));
      float huber = dist <= delta ? dist * dist / 2 :
                                    delta * dist - delta * delta / 2;
      sum_j += weights[j] * huber;
      // FIXME: Weight should be the max of i weight and j weight.
      // TODO: Don't need to compute i for j everytime.
    }
    sum += sum_j;
  }
  std::cout << "*********************" << std::endl;
  std::cout << sum << std::endl;
  std::cout << "*********************" << std::endl;
}

void WarpField::energy_reg(const std::vector<cv::Vec3f>& surface_points, const cv::Matx33f live_camera_rot, const cv::Vec3f live_camera_trans, const std::vector<cv::Matx33f>& warp_rot, const std::vector<cv::Vec3f>& warp_trans)
{
// TODO: maybe Tic and Tjc can be computed by inverse camera transform * warp?
  std::cout << "poop" << std::endl;
  cv::Matx33f z(1, 2, 3, 4, 5, 6, 7, 8, 9);
  cv::Vec3f x = warp_trans[0] - live_camera_trans;
  std::cout<< "ppp " << x << std::endl;
  cv::Affine3f T(z, x);
  // m << live_camera_rot;
  // std::cout << "1111" << std::endl;
  // std::cout << live_camera_rot.at<float>(0,0) << std::endl;
  // std::cout << "1111" << std::endl;
  // TODO: get KNN j points and for all points in j compute algorithm 8
}

// void WarpField::energy_reg(const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
//         kfusion::utils::DualQuaternion<float>>> &edges)
// {

// }


/**
 *
 * @param points
 * @param normals
 */
void WarpField::warp(std::vector<Vec3d>& points, std::vector<Vec3d>& normals) const
{
    int i = -1;
    for (auto& point : points)
    {
        i++;
        if(std::isnan(point[0]) || std::isnan(normals[i][0])) continue;
        KNN(point);
        utils::DualQuaternion<double> dqb = DQB(point);
        dqb.transform(point);
        //point = warp_to_live_ * point; //[Minhui 2018/1/28]temporarily comment this line

        dqb.transform(normals[i]);
        //normals[i] = warp_to_live_ * normals[i];  //[Minhui 2018/1/28]temporarily comment this line
    }
}

// (broken)
// void WarpField::updateTransformations(double[KNN_NEIGHBOURS][6] transformations) {
//     for(int i = 0; i < KNN_NEIGHBOURS; i++) {
//         nodes_->at(ret_index_[i]).transform.encodeRotation(transformations[i][0], transformations[i][1], transformations[i][2]);
//         nodes_->at(ret_index_[i]).transform.encodeTranslation(transformations[i][3], transformations[i][4], transformations[i][5]);
//         //TODO: (check) The function encodeTranslation() is not compatible with getTranslation() [not sure]
//     }
// }


/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
utils::DualQuaternion<double> WarpField::DQB(const Vec3f& vertex) const
{
    float weights[KNN_NEIGHBOURS];
    getWeightsAndUpdateKNN(vertex, weights);
    utils::Quaternion<double> translation_sum(0,0,0,0);
    utils::Quaternion<double> rotation_sum(0,0,0,0);
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
    {
        translation_sum += weights[i] * nodes_->at(ret_index_[i]).transform.getTranslation();
        rotation_sum += weights[i] * nodes_->at(ret_index_[i]).transform.getRotation();
    }
    //rotation_sum = utils::Quaternion<float>(); //[Minhui 2018/1/28]comment this line to avoid initializing "rotation_sum" to (1, 0, 0, 0)
    return utils::DualQuaternion<double>(translation_sum, rotation_sum);
}

/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
utils::DualQuaternion<double> WarpField::DQB_r(const Vec3d& vertex, const float weights[KNN_NEIGHBOURS], const unsigned long knn_indices_[KNN_NEIGHBOURS]) const
{
    //float weights[KNN_NEIGHBOURS];
    //getWeightsAndUpdateKNN(vertex, weights);
    utils::Quaternion<double> translation_sum(0,0,0,0);
    utils::Quaternion<double> rotation_sum(0,0,0,0);
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
    {
        translation_sum += weights[i] * nodes_->at(knn_indices_[i]).transform.getTranslation();
        rotation_sum += weights[i] * nodes_->at(knn_indices_[i]).transform.getRotation();
    }
    //rotation_sum = utils::Quaternion<float>(); //[Minhui 2018/1/28]comment this line to avoid initializing "rotation_sum" to (1, 0, 0, 0)
    return utils::DualQuaternion<double>(translation_sum, rotation_sum);
}


/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
utils::DualQuaternion<double> WarpField::DQB(const Vec3f& vertex, const std::vector<double*> epsilon) const // [Minhui 2018/1/30] (check)epsilon
{
    float weights[KNN_NEIGHBOURS];
    getWeightsAndUpdateKNN(vertex, weights);
    utils::DualQuaternion<double> eps;
    utils::Quaternion<double> translation_sum(0,0,0,0);
    utils::Quaternion<double> rotation_sum(0,0,0,0);

    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
    {
        // epsilon [0:2] is rotation [3:5] is translation
        eps.from_twist(epsilon[i][0], epsilon[i][1], epsilon[i][2],
                       epsilon[i][3], epsilon[i][4], epsilon[i][5]);

        translation_sum += weights[i] * (nodes_->at(ret_index_[i]).transform.getTranslation() + eps.getTranslation());
        rotation_sum += weights[i] * (nodes_->at(ret_index_[i]).transform.getRotation() + eps.getRotation());
    }
    rotation_sum = utils::Quaternion<double>(); //[Minhui 2018/1/28]might comment this line to avoid initializing "rotation_sum" to (1, 0, 0, 0)
    return utils::DualQuaternion<double>(translation_sum, rotation_sum);
}


/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
void WarpField::update_nodes(std::vector<double*> &epsilons)
{
    assert(epsilons.size() != 0);//if(epsilons.size() == 0) exit(1);
    utils::DualQuaternion<double> eps;
    //printf("[Debug 11]%d\n", epsilons.size());
    for(size_t j=0; j<epsilons.size(); j++)
    {
        auto epsilon = epsilons[j];
        for (size_t i = 0; i < nodes_->size(); i++)
        {
            if(std::isnan(epsilon[i*8]))
                continue;
            // epsilon [0:2] is rotation [3:5] is translation
            eps.from_twist(epsilon[i*8], epsilon[i*8 + 1], epsilon[i*8 + 2],
                        epsilon[i*8 + 4], epsilon[i*8 + 5], epsilon[i*8 + 6]);
            //printf("[Debug 12]%f %f %f %f %f %f\n", epsilon[i*8], epsilon[i*8 + 1], epsilon[i*8 + 2],
                        //epsilon[i*8 + 4], epsilon[i*8 + 5], epsilon[i*8 + 6]);
            auto tr = eps.getTranslation() + nodes_->at(i).transform.getTranslation();
            auto rot = eps.getRotation() + nodes_->at(i).transform.getRotation();
            //auto rot = utils::Quaternion<float>(); //check
            nodes_->at(i).transform = utils::DualQuaternion<double>(tr, rot);
        }
    }
}

void WarpField::update_nodes(const double *epsilon)
{
    assert(epsilon != NULL);
    utils::DualQuaternion<double> eps;
    for (size_t i = 0; i < nodes_->size(); i++)
    {
        // epsilon [0:2] is rotation [3:5] is translation
        eps.from_twist(epsilon[i*6], epsilon[i*6 +1], epsilon[i*6 + 2],
                       epsilon[i*6 + 3], epsilon[i*6 + 4], epsilon[i*6 + 5]);
        auto tr = eps.getTranslation() + nodes_->at(i).transform.getTranslation();
//        auto rot = eps.getRotation() + nodes_->at(i).transform.getRotation();
        auto rot = utils::Quaternion<double>();
        nodes_->at(i).transform = utils::DualQuaternion<double>(tr, rot);
    }
}

/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
void WarpField::getWeightsAndUpdateKNN(const Vec3f& vertex, float weights[KNN_NEIGHBOURS]) const
{
    KNN(vertex);
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
        weights[i] = weighting(out_dist_sqr_[i], nodes_->at(ret_index_[i]).weight);
}

/**
 * \brief
 * \param squared_dist
 * \param weight
 * \return
 */
float WarpField::weighting(float squared_dist, float weight) const
{
    return (float) exp(-squared_dist / (2 * weight * weight));
}

/**
 * \brief
 * \return
 */
void WarpField::KNN(Vec3f point) const
{
    resultSet_->init(&ret_index_[0], &out_dist_sqr_[0]);
    index_->findNeighbors(*resultSet_, point.val, nanoflann::SearchParams(10));
}

/**
 * \brief
 * \return
 */
const std::vector<deformation_node>* WarpField::getNodes() const
{
    return nodes_;
}

/**
 * \brief
 * \return
 */
std::vector<deformation_node>* WarpField::getNodes()
{
    return nodes_;
}

/**
 * \brief
 * \return
 */
void WarpField::buildKDTree()
{
    //    Build kd-tree with current warp nodes.
    cloud.pts.resize(nodes_->size());
    for(size_t i = 0; i < nodes_->size(); i++)
        cloud.pts[i] = nodes_->at(i).vertex;
    index_->buildIndex();
}

const cv::Mat WarpField::getNodesAsMat() const
{
    cv::Mat matrix(1, nodes_->size(), CV_32FC3);
    for(int i = 0; i < nodes_->size(); i++)
        matrix.at<cv::Vec3f>(i) = nodes_->at(i).vertex;
    return matrix;
}

/**
 * \brief
 */
void WarpField::clear()
{

}
void WarpField::setWarpToLive(const Affine3f &pose)
{
    warp_to_live_ = pose;
}

std::vector<float>* WarpField::getDistSquared() const
{
    return &out_dist_sqr_;
}

std::vector<size_t>* WarpField::getRetIndex() const
{
    return &ret_index_;
}

