#ifndef KFUSION_OPTIMISATION_H
#define KFUSION_OPTIMISATION_H
#include <cmath>
#include <dual_quaternion.hpp>
#include <kfusion/warp_field.hpp>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// [Minhui 2018/1/28] In optimisation.hpp, it defines "struct DynamicFusionDataEnergy", "struct DynamicFusionRegEnergy" and "class WarpProblem"

// template <typename T>
// struct Vec2d {
//   Vec2d() : x(0), y(0) {}
//   Vec2d(T x1, T y1) : x(x1), y(y1) {}
//   T x, y;
// };

// template <typename T>
// struct Vec3d {
//   Vec3d() : x(0), y(0), z(0) {}
//   Vec3d(T x1, T y1, T z1) : x(x1), y(y1), z(z1) {}
//   T x, y, z;
// };

struct DynamicFusionDataEnergy
{
    DynamicFusionDataEnergy(const cv::Vec3d& live_vertex,
                            const cv::Vec3d& live_normal,
                            const cv::Vec3d& canonical_vertex,
                            const cv::Vec3d& canonical_normal,
                            kfusion::WarpField *warpField,
                            const float weights[KNN_NEIGHBOURS],
                            const unsigned long knn_indices[KNN_NEIGHBOURS],
                            const kfusion::Intr& intr,
                            const int& index)
            : live_vertex_(live_vertex),
              live_normal_(live_normal),
              canonical_vertex_(canonical_vertex),
              canonical_normal_(canonical_normal),
              warpField_(warpField),
              intr_(intr),
              index_(index)
    {
        weights_ = new float[KNN_NEIGHBOURS];
        knn_indices_ = new unsigned long[KNN_NEIGHBOURS];
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
        {
            weights_[i] = weights[i];
            knn_indices_[i] = knn_indices[i];
        }
    }
    ~DynamicFusionDataEnergy()
    {
        delete[] weights_;
        delete[] knn_indices_;
    }

    // [Minhui 2018/1/28] original
    // template <typename T>
    // bool operator()(T const * const * epsilon_, T* residuals) const
    // {
    //     //printf("(In DynamicFusionDataEnergy)\n");
    //     auto nodes = warpField_->getNodes();

    //     T total_translation[3] = {T(0), T(0), T(0)};
    //     float total_translation_float[3] = {0, 0, 0};
    //     T total_quaternion[4] = {T(0), T(0), T(0), T(0)};

    //     for(int i = 0; i < KNN_NEIGHBOURS; i++)
    //     {
    //         auto quat = nodes->at(knn_indices_[i]).transform;
    //         cv::Vec3f vert;
    //         quat.getTranslation(vert);

    //         T eps_t[3] = {epsilon_[i][3], epsilon_[i][4], epsilon_[i][5]};

    //         float temp[3];
    //         quat.getTranslation(temp[0], temp[1], temp[2]);

    //         total_translation[0] += (T(temp[0]) +  eps_t[0]) * T(weights_[i]);
    //         total_translation[1] += (T(temp[1]) +  eps_t[1]) * T(weights_[i]);
    //         total_translation[2] += (T(temp[2]) +  eps_t[2]) * T(weights_[i]);

    //     }

    //     T norm = ceres::sqrt(total_translation[0] * total_translation[0] +
    //                          total_translation[1] * total_translation[1] +
    //                          total_translation[2] * total_translation[2]);

    //     residuals[0] = T(live_vertex_[0] - canonical_vertex_[0]) + total_translation[0];
    //     residuals[1] = T(live_vertex_[1] - canonical_vertex_[1]) + total_translation[1];
    //     residuals[2] = T(live_vertex_[2] - canonical_vertex_[2]) + total_translation[2];

    //     return true;
    // }
 
    bool operator()(const double* const* epsilon_, double* residuals) const
    {
        auto nodes = warpField_->getNodes();
        cv::Vec3d canonical_point = canonical_vertex_;
        cv::Vec3d canonical_point_n = canonical_normal_;

        // [Step 2] Calculte the dqb warp function of the canonical point
        // Reference to warpField_->DQB_r
        kfusion::utils::Quaternion<double> translation_sum(0.0, 0.0, 0.0, 0.0);
        kfusion::utils::Quaternion<double> rotation_sum(0.0, 0.0, 0.0, 0.0);

        float weights_sum = 0;
        for(int i = 0; i < KNN_NEIGHBOURS; i++) {
            weights_sum += weights_[i];
        }
        for(int i = 0; i < KNN_NEIGHBOURS; i++) {
            weights_[i] = weights_[i] / weights_sum;
        }

        for(int i = 0; i < KNN_NEIGHBOURS; i++) {
            // double a = 9;
            // printf("[1] %f %f %f %f\n", epsilon_[i][0], epsilon_[i][1], epsilon_[i][2], epsilon_[i][3]);
            // printf("[2] %f %f %f %f\n", nodes->at(knn_indices_[i]).transform.rotation_.w_, nodes->at(knn_indices_[i]).transform.rotation_.x_, nodes->at(knn_indices_[i]).transform.rotation_.y_, nodes->at(knn_indices_[i]).transform.rotation_.z_);
            // printf("[1] %f %f %f %f\n", epsilon_[i][4], epsilon_[i][5], epsilon_[i][6], epsilon_[i][7]);
            // printf("[2] %f %f %f %f\n\n", nodes->at(knn_indices_[i]).transform.translation_.w_, nodes->at(knn_indices_[i]).transform.translation_.x_, nodes->at(knn_indices_[i]).transform.translation_.y_, nodes->at(knn_indices_[i]).transform.translation_.z_);
            kfusion::utils::DualQuaternion<double> temp;
            kfusion::utils::Quaternion<double> rotation(0.0f, 0.0f, 0.0f, 0.0f);
            kfusion::utils::Quaternion<double> translation(0.0f, 0.0f, 0.0f, 0.0f);
            temp.rotation_.w_ = epsilon_[i][0];
            temp.rotation_.x_ = epsilon_[i][1];
            temp.rotation_.y_ = epsilon_[i][2];
            temp.rotation_.z_ = epsilon_[i][3];
            temp.translation_.w_ = epsilon_[i][4];
            temp.translation_.x_ = epsilon_[i][5];
            temp.translation_.y_ = epsilon_[i][6];
            temp.translation_.z_ = epsilon_[i][7];

            rotation = temp.getRotation();
            translation = temp.getTranslation();

            rotation_sum += weights_[i] * rotation;
            translation_sum += weights_[i] * translation;
        }
        // kfusion::utils::DualQuaternion<double> dqb;
        // dqb.from_twist(rotation_sum.x_, rotation_sum.y_, rotation_sum.z_,
        //                     translation_sum.x_, translation_sum.y_, translation_sum.z_);
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

        // dqb.encodeRotation(rotation_sum.w_, rotation_sum.x_, rotation_sum.y_, rotation_sum.z_);
        // dqb.encodeTranslation(translation_sum.x_, translation_sum.y_, translation_sum.z_);


        // [Step 2] project the 3D conanical point to live-frame image domain (2D-coordinate)
        cv::Vec2d project_point = project(canonical_point[0], canonical_point[1], canonical_point[2]); //Is point.z needs to be in homogeneous coordinate? (point.z==1)
        double project_u = project_point[0];
        double project_v = project_point[1];
        double depth = 0.0f;

        if(project_u >= 480 || project_u < 0 || project_v >= 640 || project_v < 0 ) {
            residuals[0] = 1000000000;
            residuals[1] = 1000000000;
            residuals[2] = 1000000000;
            return true;
        }

        depth = warpField_->live_vertices_[project_u * 640 + project_v][2];
        
        if(std::isnan(depth)) {
            residuals[0] = 1000000000;
            residuals[1] = 1000000000;
            residuals[2] = 1000000000;
            return true;
        }

        // [Step 3] re-project the correspondence to 3D space
        cv::Vec3d reproject_point = reproject(project_u, project_v, depth);
        double reproject_x = reproject_point[0];
        double reproject_y = reproject_point[1];
        double reproject_z = reproject_point[2];

        // // [Step 4] Calculate the residual
        // residuals[0] = canonical_point_n[0] * (canonical_point[0] - reproject_x);
        // residuals[1] = canonical_point_n[1] * (canonical_point[1] - reproject_y);
        // residuals[2] = canonical_point_n[2] * (canonical_point[2] - reproject_z);

        // Not multiply the normal vector of warped canonical point
        residuals[0] = canonical_point[0] - reproject_x;
        residuals[1] = canonical_point[1] - reproject_y;
        residuals[2] = canonical_point[2] - reproject_z;

        // residuals[0] = sqrt(pow((canonical_point[0] - reproject_x), 2) +
        //                 pow((canonical_point[1] - reproject_y), 2) +
        //                 pow((canonical_point[2] - reproject_z), 2));

        //printf("*****[%d] %f %f %f\n", index_, residuals[0], residuals[1], residuals[2]);
     
        if(std::isnan(residuals[0]) || std::isnan(residuals[1]) || std::isnan(residuals[2])) {
            cv::waitKey(0);
        }

        return true;
    }

    cv::Vec2d project(const double &x, const double &y, const double &z) const {
        cv::Vec2d project_point;

        project_point[0] = intr_.fx * (x / z) + intr_.cx;
        project_point[1] = intr_.fy * (y / z) + intr_.cy;

        return project_point;
    }

    cv::Vec3d reproject(const double &u, const double &v, const double &depth) const {
        cv::Vec3d reproject_point;

        reproject_point[0] = depth * (u - intr_.cx) * intr_.fx;
        reproject_point[1] = depth * (v - intr_.cy) * intr_.fy;
        reproject_point[2] = depth;
    }

// template <typename T>
// struct Vec3d {
//   Vec3d() : x(0), y(0), z(0) {}
//   Vec3d(T x1, T y1, T z1) : x(x1), y(y1), z(z1) {}
//   T x, y, z;
// };

    // float2 project(const float &x, const float &y, const float &z) const
    // {
    //     float2 project_point;

    //     project_point.x = intr_.fx * (x / z) + intr_.cx;
    //     project_point.y = intr_.fy * (y / z) + intr_.cy;

    //     return project_point;
    // }

    // float3 reproject(const float &u, const float &v, const float &depth) const
    // {
    //     float3 reproject_point;
        
    //     reproject_point.x = depth * (u - intr_.cx) * intr_.fx;
    //     reproject_point.y = depth * (v - intr_.cy) * intr_.fy;
    //     reproject_point.z = depth;

    //     return reproject_point;
    // }


/**
 * Tukey loss function as described in http://web.as.uky.edu/statistics/users/pbreheny/764-F11/notes/12-1.pdf
 * \param x
 * \param c
 * \return
 *
 * \note
 * The value c = 4.685 is usually used for this loss function, and
 * it provides an asymptotic efficiency 95% that of linear
 * regression for the normal distribution
 *
 * In the paper, a value of 0.01 is suggested for c
 */
    template <typename T>
    T tukeyPenalty(T x, T c = T(0.01)) const
    {
        //TODO: this seems to mean that 0.01 is the acceptable threshold for x (otherwise return 0 and as such, it converges). Need to check if this is correct
        return ceres::abs(x) <= c ? x * ceres::pow((T(1.0) - (x * x) / (c * c)), 2) : T(0.0);
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
//      TODO: this will only have one residual at the end, remember to change
    static ceres::CostFunction* Create(const cv::Vec3d& live_vertex,
                                       const cv::Vec3d& live_normal,
                                       const cv::Vec3d& canonical_vertex,
                                       const cv::Vec3d& canonical_normal,
                                       kfusion::WarpField* warpField,
                                       const float weights[KNN_NEIGHBOURS],
                                       const unsigned long ret_index[KNN_NEIGHBOURS],
                                       const kfusion::Intr& intr,
                                       const int index)
    {
        auto cost_function = new ceres::DynamicNumericDiffCostFunction<DynamicFusionDataEnergy, ceres::CENTRAL>(
                new DynamicFusionDataEnergy(live_vertex,
                                            live_normal,
                                            canonical_vertex,
                                            canonical_normal,
                                            warpField,
                                            weights,
                                            ret_index,
                                            intr,
                                            index));
        for(int i=0; i < KNN_NEIGHBOURS; i++)
            cost_function->AddParameterBlock(8);
        cost_function->SetNumResiduals(3);
        return cost_function;
    }
    const cv::Vec3d live_vertex_;
    const cv::Vec3d live_normal_;
    const cv::Vec3d canonical_vertex_;
    const cv::Vec3d canonical_normal_;
    const kfusion::Intr& intr_;

    float *weights_;
    unsigned long *knn_indices_;
    int index_;

    kfusion::WarpField *warpField_;
};

struct DynamicFusionRegEnergy {
  DynamicFusionRegEnergy(const std::vector<kfusion::deformation_node> &nodes,
                         const std::vector<size_t> &ret_index,
                         const float weights[KNN_NEIGHBOURS],
                         const cv::Affine3f &inverse_pose)
      : ret_index_(ret_index), inverse_pose_(inverse_pose) {
    nodes_ = &nodes;
    weights_ = new float[KNN_NEIGHBOURS];
    for (int i = 0; i < KNN_NEIGHBOURS; i++) {
      weights_[i] = weights[i];
    }
  };

  virtual ~DynamicFusionRegEnergy(){
    delete[] weights_;
  };

  bool operator()(double const *const *epsilon_, double *residuals) const {
    double sum_j = 0;
    double delta = 0.0001;
    kfusion::utils::Quaternion<double> rotation_sum(0.0f, 0.0f, 0.0f, 0.0f);
    kfusion::utils::Quaternion<double> translation_sum(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < KNN_NEIGHBOURS; ++i) {
      kfusion::utils::DualQuaternion<double> temp;
      kfusion::utils::Quaternion<double> rotation(0.0f, 0.0f, 0.0f, 0.0f);
      kfusion::utils::Quaternion<double> translation(0.0f, 0.0f, 0.0f, 0.0f);
      temp.encodeRotation(epsilon_[i][0], epsilon_[i][1], epsilon_[i][2]);
      temp.encodeTranslation(epsilon_[i][3], epsilon_[i][4], epsilon_[i][5]);
      rotation = temp.getRotation();
      translation = temp.getTranslation();
      rotation_sum += weights_[i] * rotation;
      translation_sum += weights_[i] * translation;
    }
    double sinr = 2.0 * (rotation_sum.w_ * rotation_sum.x_ +
                         rotation_sum.y_ * rotation_sum.z_);
    double cosr = 1.0 - 2.0 * (pow(rotation_sum.x_, 2) + pow(rotation_sum.y_, 2));
    double siny = 2.0 * (rotation_sum.w_ * rotation_sum.z_ +
                         rotation_sum.x_ * rotation_sum.y_);
    double cosy = 1.0 - 2.0 * (rotation_sum.y_ * rotation_sum.y_ +
                               rotation_sum.z_ * rotation_sum.z_);
    double sinp = 2.0 * (rotation_sum.w_ * rotation_sum.y_ - rotation_sum.z_ * rotation_sum.x_);
    cv::Vec3f i_rot;
	if (fabs(sinp) >= 1)
        cv::Vec3f i_rot(atan2(sinr, cosr), copysign(M_PI / 2, sinp), atan2(siny, cosy));
	else
        cv::Vec3f i_rot(atan2(sinr, cosr), asin(sinp), atan2(siny, cosy));
    cv::Vec3f i_trans(translation_sum.x_, translation_sum.y_, translation_sum.z_);
    cv::Affine3f i_warp(i_rot, i_trans);
    cv::Affine3f Tic;
    Tic = inverse_pose_.concatenate(i_warp);
    for (int i = 0; i < KNN_NEIGHBOURS; ++i) {
      auto j_point = nodes_->at(ret_index_[i]).vertex;
      cv::Vec3f j_rot(epsilon_[i][0], epsilon_[i][1], epsilon_[i][2]);
      cv::Vec3f j_trans(epsilon_[i][3], epsilon_[i][4], epsilon_[i][5]);
      cv::Affine3f j_warp(j_rot, j_trans);
      cv::Affine3f Tjc;
      Tjc = inverse_pose_.concatenate(j_warp);
      auto difference = Tic * j_point - Tjc * j_point;
      float dist = sqrt(pow(difference[0], 2) +
                        pow(difference[1], 2) +
                        pow(difference[2], 2));
      float huber = dist <= delta ? dist * dist / 2 :
                                    delta * dist - delta * delta / 2;
      sum_j += weights_[i] * huber;
    }
    residuals[0] = sum_j;
    return true;
  }

  static ceres::CostFunction *Create(
      const std::vector<kfusion::deformation_node> &nodes,
      const std::vector<size_t> &ret_index,
      const float weights[KNN_NEIGHBOURS],
      const cv::Affine3f &inverse_pose) {
    auto j_point = nodes.at(ret_index[0]).vertex;
    auto cost_function =
        new ceres::DynamicNumericDiffCostFunction<DynamicFusionRegEnergy>(
            // new DynamicFusionRegEnergy(ret_index, weights,
            new DynamicFusionRegEnergy(nodes, ret_index, weights,
                                       inverse_pose));
    for (int i = 0; i < KNN_NEIGHBOURS; i++)
      cost_function->AddParameterBlock(8);
    cost_function->SetNumResiduals(1);
    // delete DynamicFusionRegEnergy;
    return cost_function;
  }

  const std::vector<kfusion::deformation_node> *nodes_;
  const std::vector<size_t> ret_index_;
  float *weights_;
  cv::Affine3f inverse_pose_;
};

class WarpProblem {
// <<<<<<< HEAD
//  public:
//   explicit WarpProblem(kfusion::WarpField *warp) : warpField_(warp) {
//     parameters_.resize(warpField_->getNodes()->size() * 8);
//     for(int i = 0; i < warpField_->getNodes()->size(); i++) {
//       parameters_[i * 8 + 0] = &(warpField_->getNodes()->at(i).transform.rotation_.w_);
//       parameters_[i * 8 + 1] = &(warpField_->getNodes()->at(i).transform.rotation_.x_);
//       parameters_[i * 8 + 2] = &(warpField_->getNodes()->at(i).transform.rotation_.y_);
//       parameters_[i * 8 + 3] = &(warpField_->getNodes()->at(i).transform.rotation_.z_);
//       parameters_[i * 8 + 4] = &(warpField_->getNodes()->at(i).transform.translation_.w_);
//       parameters_[i * 8 + 5] = &(warpField_->getNodes()->at(i).transform.translation_.x_);
//       parameters_[i * 8 + 6] = &(warpField_->getNodes()->at(i).transform.translation_.y_);
//       parameters_[i * 8 + 7] = &(warpField_->getNodes()->at(i).transform.translation_.z_);
//     }

//     // // TODO: Retrieved type of "rotation" needs to be check (x, y, z) or
//     // (angle, x, y, z)?
//   };
// =======
  public:
    explicit WarpProblem(kfusion::WarpField *warp) : warpField_(warp)
    {
        parameters_ = new double[warpField_->getNodes()->size() * 8];
        for(int i = 0; i < warpField_->getNodes()->size(); i++) {
            parameters_[i * 8 + 0] = warpField_->getNodes()->at(i).transform.rotation_.w_;
            parameters_[i * 8 + 1] = warpField_->getNodes()->at(i).transform.rotation_.x_;
            parameters_[i * 8 + 2] = warpField_->getNodes()->at(i).transform.rotation_.y_;
            parameters_[i * 8 + 3] = warpField_->getNodes()->at(i).transform.rotation_.z_;
            parameters_[i * 8 + 4] = warpField_->getNodes()->at(i).transform.translation_.w_;
            parameters_[i * 8 + 5] = warpField_->getNodes()->at(i).transform.translation_.x_;
            parameters_[i * 8 + 6] = warpField_->getNodes()->at(i).transform.translation_.y_;
            parameters_[i * 8 + 7] = warpField_->getNodes()->at(i).transform.translation_.z_;
        }
    };

    ~WarpProblem() {
        delete[] parameters_;
    }

    std::vector<double*> mutable_epsilon(const unsigned long *index_list) const
    {
        std::vector<double*> mutable_epsilon_(KNN_NEIGHBOURS);
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
            mutable_epsilon_[i] = &(parameters_[index_list[i] * 8]); // Blocks of 8
        return mutable_epsilon_;
    }

    std::vector<double*> mutable_epsilon(const std::vector<size_t>& index_list) const
    {
        std::vector<double*> mutable_epsilon_(KNN_NEIGHBOURS);
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
            mutable_epsilon_[i] = &(parameters_[index_list[i] * 8]); // Blocks of 8
        return mutable_epsilon_;
    }
    double *mutable_params()
    {
        return parameters_;
    }

    const double *params() const
    {
        return parameters_;
    }
// >>>>>>> minhui

  // ~WarpProblem() {
  // }

// <<<<<<< HEAD
//   std::vector<double *> mutable_epsilon(
//       const unsigned long *index_list) const {
//     std::vector<double *> mutable_epsilon_(KNN_NEIGHBOURS);
//     for (int i = 0; i < KNN_NEIGHBOURS; i++) {
//       // mutable_epsilon_[i] =
//       // &(nodes_->at(index_list[i]).transform.translation_.x_);
//       mutable_epsilon_[i] = parameters_[index_list[i] * 8];
//     }
//     return mutable_epsilon_;
//   }

//   std::vector<double *> mutable_epsilon(
//       const std::vector<size_t> &index_list) const {
//     std::vector<double *> mutable_epsilon_(KNN_NEIGHBOURS);
//     for (int i = 0; i < KNN_NEIGHBOURS; i++)
//       mutable_epsilon_[i] = parameters_[index_list[i] * 8];  // Blocks of 8
//     return mutable_epsilon_;
//   }

//   // double *mutable_params()
//   // {
//   //     return parameters_;
//   // }

//   // YuYang
//   // const double *params() const
//   const std::vector<double *> *params() const {
//     // return parameters_;
//     return &parameters_;
//   }

//  private:
//   // double *parameters_;
//   std::vector<double *> parameters_;
//   kfusion::WarpField *warpField_;
// };

  private:
    double *parameters_;
    kfusion::WarpField *warpField_;
};


#endif //KFUSION_OPTIMISATION_H
