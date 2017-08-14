#pragma once

#include <kfusion/types.hpp>
#include <dual_quaternion.hpp>

namespace kfusion
{
    class WarpField;
    namespace cuda
    {
        class KF_EXPORTS TsdfVolume
        {
        public:
            TsdfVolume(const cv::Vec3i& dims);
            virtual ~TsdfVolume();

            void create(const Vec3i& dims);

            Vec3i getDims() const;
            Vec3f getVoxelSize() const;

            const CudaData data() const;
            CudaData data();

            cv::Mat get_cloud_host() const;
            cv::Mat get_normal_host() const;

            Vec3f getSize() const;
            void setSize(const Vec3f& size);

            float getTruncDist() const;
            void setTruncDist(float distance);

            int getMaxWeight() const;
            void setMaxWeight(int weight);

            Affine3f getPose() const;
            void setPose(const Affine3f& pose);

            float getRaycastStepFactor() const;
            void setRaycastStepFactor(float factor);

            float getGradientDeltaFactor() const;
            void setGradientDeltaFactor(float factor);

            Vec3i getGridOrigin() const;
            void setGridOrigin(const Vec3i& origin);

            float psdf(Vec3f voxel_center, Vec3f warped, const Dists& depth_img, const Intr& intr);
            float weighting(const std::vector<float>& dist_sqr, int k) const;
            void compute_tsdf_value(Vec3f vertex, Vec3f voxel_center, float weight);
            void surface_fusion(const WarpField& warp_field,
                                const cuda::Dists& depth_img,
                                const Affine3f& camera_pose,
                                const Intr& intr);
            virtual void clear();
            virtual void applyAffine(const Affine3f& affine);
            virtual void integrate(const Dists& dists, const Affine3f& camera_pose, const Intr& intr);
            virtual void raycast(const Affine3f& camera_pose, const Intr& intr, Depth& depth, Normals& normals);
            virtual void raycast(const Affine3f& camera_pose, const Intr& intr, Cloud& points, Normals& normals);

            void swap(CudaData& data);

            DeviceArray<Point> fetchCloud(DeviceArray<Point>& cloud_buffer) const;
            void fetchNormals(const DeviceArray<Point>& cloud, DeviceArray<Normal>& normals) const;
            void compute_points();
            void compute_normals() ;


        private:
            CudaData data_;
//            need to make this smart pointers
            cuda::DeviceArray<Point> *cloud_buffer;
            cuda::DeviceArray<Point> *cloud;
            cuda::DeviceArray<Normal> *normal_buffer;
            cv::Mat *cloud_host;
            cv::Mat *normal_host;

            float trunc_dist_;
            int max_weight_;
            Vec3i dims_;
            Vec3f size_;
            Affine3f pose_;
            float gradient_delta_factor_;
            float raycast_step_factor_;
            // TODO: remember to add entry when adding a new node
            struct Entry
            {
                float tsdf_value;
                float tsdf_weight;
            };

            std::vector<Entry> tsdf_entries;
        };
    }
}
