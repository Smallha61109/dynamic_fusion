#ifndef KFUSION_OPTIMISATION_H
#define KFUSION_OPTIMISATION_H
#include <dual_quaternion.hpp>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <kfusion/warp_field.hpp>

// [Minhui 2018/1/28] In optimisation.hpp, it defines "struct DynamicFusionDataEnergy", "struct DynamicFusionRegEnergy" and "class WarpProblem"

struct DynamicFusionDataEnergy
{
    // DynamicFusionDataEnergy(const std::vector<cv::Vec3f>& live_vertex,
    //                         const std::vector<cv::Vec3f>& live_normal,
    //                         const cv::Vec3f& canonical_vertex,
    //                         const cv::Vec3f& canonical_normal,
    //                         kfusion::WarpField *warpField,
    //                         const float weights[KNN_NEIGHBOURS],
    //                         const unsigned long knn_indices[KNN_NEIGHBOURS],
    //                         const kfusion::Intr& intr)
    DynamicFusionDataEnergy(const cv::Vec3d& live_vertex,
                            const cv::Vec3d& live_normal,
                            const cv::Vec3d& canonical_vertex,
                            const cv::Vec3d& canonical_normal,
                            kfusion::WarpField *warpField,
                            const float weights[KNN_NEIGHBOURS],
                            const unsigned long knn_indices[KNN_NEIGHBOURS],
                            const kfusion::Intr& intr)
            : live_vertex_(live_vertex),
              live_normal_(live_normal),
              canonical_vertex_(canonical_vertex),
              canonical_normal_(canonical_normal),
              warpField_(warpField),
              intr_(intr)
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

    // [Minhui 2018/1/28] 
    template <typename T>
    bool operator()(T const * const * epsilon_, T* residuals) const
    {
        auto nodes = warpField_->getNodes();
        cv::Vec3d canonical_point = canonical_vertex_;
        cv::Vec3d canonical_point_n = canonical_normal_;

        // [Step 1] The point in canonical model warps using warp functiton (3D-coordinate)
        if(std::isnan(canonical_point[0]) && std::isnan(canonical_point_n[0])) {
            return false; //[Minhui 2018/1/28] not sure if it is ok to return false
        }

        // [Minhui 2018/1/29]update transformations in KNNs
        // double KNNtranformations[KNN_NEIGHBOURS][6];
        // for(int i = 0; i < KNN_NEIGHBOURS; i++) {
        //     for(int j = 0; j < 6; j++) {
                // ceres::Jet<4> result;
                // (*functor_)(epsilon_, &result);
                // KNNtransromations[i][j] = result[i][j];
               //KNNtranformations[i][j] = epsilon_[i][j];
            //    if (epsilon_[i][j] >= 0) {
            //         KNNtranformations[KNN_NEIGHBOURS][j] = ceres::abs(epsilon_[i][j]);
            //    }
            //    else {
            //         KNNtranformations[KNN_NEIGHBOURS][j] = -ceres::abs(epsilon_[i][j]);
            //    }
               //KNNtranformations[KNN_NEIGHBOURS][j] = ceres::abs(epsilon_[i][j]);
               //KNNtranformations[KNN_NEIGHBOURS][j] = epsilon_[i][j];
               //const ceres::Jet<double, 4>* test = epsilon_[i];
            //}
            //nodes->at(knn_indices_[i]).transform.encodeRotation(epsilon_[i][0], epsilon_[i][1], epsilon_[i][2]);
            //nodes->at(knn_indices_[i]).transform.encodeTranslation(epsilon_[i][3], epsilon_[i][4], epsilon_[i][5]);
            //nodes->at(knn_indices_[i]).transform.encodeRotation(KNNtranformations[i][0], KNNtranformations[i][1], KNNtranformations[i][2]);
            //nodes->at(knn_indices_[i]).transform.encodeTranslation(KNNtranformations[i][3], KNNtranformations[i][4], KNNtranformations[i][5]);
        //}
        //warpField_->updateTransformations(KNNtranformations);

        // printf("[Debug e]%f %f %f %f %f %f\n", nodes->at(knn_indices_[1]).transform.rotation_.x_
        // , nodes->at(knn_indices_[1]).transform.rotation_.y_
        // , nodes->at(knn_indices_[1]).transform.rotation_.z_
        // , nodes->at(knn_indices_[1]).transform.translation_.x_
        // , nodes->at(knn_indices_[1]).transform.translation_.y_
        // , nodes->at(knn_indices_[1]).transform.translation_.z_ );
        kfusion::utils::DualQuaternion<double> dqb = warpField_->DQB_r(canonical_point, weights_, knn_indices_);
        dqb.transform(canonical_point);
        dqb.transform(canonical_point_n);

        // [Step 2] project the 3D conanical point to live-frame image domain (2D-coordinate)
        float2 project_point = project(canonical_point[0], canonical_point[1], canonical_point[2]);
        //is point.z needs to be in homogeneous coordinate? (point.z==1)
        float project_u = project_point.x;
        float project_v = project_point.y;
        float depth = 0.0f;

        std::vector<cv::Vec3d> live_vertices;
        live_vertices =  warpField_->live_vertices_;
        depth = live_vertices[project_u * 640 + project_v][2];

        // [Step 3] re-project the correspondence to 3D space
        float3 reproject_point = reproject(project_u, project_v, depth);
        float reproject_x = reproject_point.x;
        float reproject_y = reproject_point.y;
        float reproject_z = reproject_point.z;
        
        // [Step 4] Calculate the residual
        T residual_temp = T( canonical_point_n[0] * (canonical_point[0] - reproject_x) +
                             canonical_point_n[1] * (canonical_point[1] - reproject_y) +
                             canonical_point_n[2] * (canonical_point[2] - reproject_z));
        
        residuals[0] = tukeyPenalty(residual_temp);
        //residuals[0] = residual_temp;
        return true;
    }  

    float2 project(const float &x, const float &y, const float &z) const
    {
        float2 project_point;

        project_point.x = intr_.fx * (x / z) + intr_.cy;
        project_point.x = intr_.fy * (y / z) + intr_.cy;

        return project_point;
    }

    float3 reproject(const float &u, const float &v, const float &depth) const
    {
        float3 reproject_point;
        
        reproject_point.x = depth * (u - intr_.cx) * intr_.fx;
        reproject_point.y = depth * (v - intr_.cy) * intr_.fy;
        reproject_point.z = depth;

        return reproject_point;
    }

    template <typename T>
    double T_to_double(T a) const
    {
        return a;
    }
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
    // static ceres::CostFunction* Create(const std::vector<cv::Vec3f>* live_vertex,
    //                                    const std::vector<cv::Vec3f>* live_normal,
    //                                    const cv::Vec3f& canonical_vertex,
    //                                    const cv::Vec3f& canonical_normal,
    //                                    kfusion::WarpField* warpField,
    //                                    const float weights[KNN_NEIGHBOURS],
    //                                    const unsigned long ret_index[KNN_NEIGHBOURS],
    //                                    const kfusion::Intr& intr)
    static ceres::CostFunction* Create(const cv::Vec3f& live_vertex,
                                       const cv::Vec3f& live_normal,
                                       const cv::Vec3f& canonical_vertex,
                                       const cv::Vec3f& canonical_normal,
                                       kfusion::WarpField* warpField,
                                       const float weights[KNN_NEIGHBOURS],
                                       const unsigned long ret_index[KNN_NEIGHBOURS],
                                       const kfusion::Intr& intr)
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<DynamicFusionDataEnergy, 4>(
                new DynamicFusionDataEnergy(live_vertex,
                                            live_normal,
                                            canonical_vertex,
                                            canonical_normal,
                                            warpField,
                                            weights,
                                            ret_index,
                                            intr));
        for(int i=0; i < KNN_NEIGHBOURS; i++)
            cost_function->AddParameterBlock(6);
        //cost_function->SetNumResiduals(3); // [Minhui 2018/1/28]original
        cost_function->SetNumResiduals(1); // [Minhui 2018/1/28]modified
        return cost_function;
    }
    const cv::Vec3f live_vertex_;
    const cv::Vec3f live_normal_;
    //const std::vector<cv::Vec3f> live_vertex_;
    //const std::vector<cv::Vec3f> live_normal_;
    const cv::Vec3f canonical_vertex_;
    const cv::Vec3f canonical_normal_;
    const kfusion::Intr& intr_;

    float *weights_;
    unsigned long *knn_indices_;

    kfusion::WarpField *warpField_;
};

struct DynamicFusionRegEnergy
{
    DynamicFusionRegEnergy(){};
    ~DynamicFusionRegEnergy(){};
    template <typename T>
    bool operator()(T const * const * epsilon_, T* residuals) const
    {
        return true;
    }

/**
 * Huber penalty function, implemented as described in https://en.wikipedia.org/wiki/Huber_loss
 * In the paper, a value of 0.0001 is suggested for delta.
 * \param a
 * \param delta
 * \return
 */
    template <typename T>
    T huberPenalty(T a, T delta = 0.0001) const
    {
        return ceres::abs(a) <= delta ? a * a / 2 : delta * ceres::abs(a) - delta * delta / 2;
    }

    static ceres::CostFunction* Create()
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<DynamicFusionRegEnergy, 4>(
                new DynamicFusionRegEnergy());
        for(int i=0; i < KNN_NEIGHBOURS; i++)
            cost_function->AddParameterBlock(6);
        cost_function->SetNumResiduals(3);
        return cost_function;
    }
};

class WarpProblem {
public:
    explicit WarpProblem(kfusion::WarpField *warp) : warpField_(warp)
    {
        parameters_.resize(warpField_->getNodes()->size() * 6);
        for(int i = 0; i < warpField_->getNodes()->size(); i++) {
            parameters_[i * 6 + 0] = &(warpField_->getNodes()->at(i).transform.rotation_.x_);
            parameters_[i * 6 + 1] = &(warpField_->getNodes()->at(i).transform.rotation_.y_);
            parameters_[i * 6 + 2] = &(warpField_->getNodes()->at(i).transform.rotation_.z_);
            parameters_[i * 6 + 3] = &(warpField_->getNodes()->at(i).transform.translation_.x_);
            parameters_[i * 6 + 4] = &(warpField_->getNodes()->at(i).transform.translation_.y_);
            parameters_[i * 6 + 5] = &(warpField_->getNodes()->at(i).transform.translation_.z_);
        }

        // mutable_epsilon_[i] = &(nodes_->at(index_list[i]).transform.translation_.x_);

        // parameters_ = new double[warpField_->getNodes()->size() * 6];
        // // [Minhui 2018/1/29] Initialization per frame (update the parameters_ with wrap functions in last frame)
        // // TODO: Retrieved type of "rotation" needs to be check (x, y, z) or (angle, x, y, z)?
        // for(int i = 0; i < warpField_->getNodes()->size(); i++) {
        //     //double translation[3];
        //     cv::Vec3f translation;

        //     warpField_->getNodes()->at(i).transform.getTranslation(translation);
        //     auto rotation = warpField_->getNodes()->at(i).transform.getRotation();
        //     parameters_[i * 6 + 0] = rotation.x_;
        //     parameters_[i * 6 + 1] = rotation.y_;
        //     parameters_[i * 6 + 2] = rotation.z_;
        //     parameters_[i * 6 + 3] = translation[0];
        //     parameters_[i * 6 + 4] = translation[1];
        //     parameters_[i * 6 + 5] = translation[2];        
        // }
        // parameters_ = new double[warpField_->getNodes()->size() * 6];

    };

    ~WarpProblem() {
        //delete[] parameters_;
        std::vector<double*> *v = &parameters_;
        delete v;
    }
    std::vector<double*> mutable_epsilon(const unsigned long *index_list) const
    {
        std::vector<double*> mutable_epsilon_(KNN_NEIGHBOURS);
        for(int i = 0; i < KNN_NEIGHBOURS; i++) {
            //mutable_epsilon_[i] = &(nodes_->at(index_list[i]).transform.translation_.x_);
            mutable_epsilon_[i] = parameters_[index_list[i] * 6];
        }
        return mutable_epsilon_;
    }

    // std::vector<double*> mutable_epsilon(const std::vector<size_t>& index_list) const
    // {
    //     std::vector<double*> mutable_epsilon_(KNN_NEIGHBOURS);
    //     for(int i = 0; i < KNN_NEIGHBOURS; i++)
    //         mutable_epsilon_[i] = &(parameters_[index_list[i] * 6]); // Blocks of 6
    //     return mutable_epsilon_;
    // }

    // double *mutable_params()
    // {
    //     return parameters_;
    // }

    //YuYang
    // const double *params() const
    const std::vector<double*> *params() const
    {
        // return parameters_;
        return &parameters_;
    }


private:
    //double *parameters_;
    std::vector<double*> parameters_;
    kfusion::WarpField *warpField_;
};

#endif //KFUSION_OPTIMISATION_H
