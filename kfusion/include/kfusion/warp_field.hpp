#ifndef KFUSION_WARP_FIELD_HPP
#define KFUSION_WARP_FIELD_HPP

#include <dual_quaternion.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/types.hpp>
#include <knn_point_cloud.hpp>
#include <nanoflann/nanoflann.hpp>
#define KNN_NEIGHBOURS 8

namespace kfusion {
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, utils::PointCloud>, utils::PointCloud, 3 /* dim */
    >
    kd_tree_t;

/*!
 * \struct node
 * \brief A node of the warp field
 * \details The state of the warp field Wt at time t is defined by the values
 * of a set of n
 * deformation nodes Nt_warp = {dg_v, dg_w, dg_se3}_t. Here, this is
 * represented as follows
 *
 * \var node::vertex
 * Position of the vertex in space. This will be used when computing KNN for
 * warping points.
 *
 * \var node::transform
 * Transformation for each vertex to warp it into the live frame, equivalent to
 * dg_se in the paper.
 *
 * \var node::weight
 * Equivalent to dg_w
 */
struct deformation_node {
  Vec3f vertex;
  kfusion::utils::DualQuaternion<double> transform;
  double weight = 0;
};

class WarpField {
 public:
  WarpField();
  ~WarpField();

  void init(const cv::Mat& first_frame);
  void init(const std::vector<Vec3f>& first_frame);
  void energy(
      const cuda::Cloud& frame, const cuda::Normals& normals,
      const Affine3f& pose, const cuda::TsdfVolume& tsdfVolume,
      const std::vector<std::pair<kfusion::utils::DualQuaternion<double>,
                                  kfusion::utils::DualQuaternion<double>>>&
          edges);

  void energy_data(const std::vector<Vec3d>& canonical_vertices,
                   const std::vector<Vec3d>& canonical_normals,
                   const std::vector<Vec3d>& live_vertices,
                   const std::vector<Vec3d>& live_normals, const Intr& intr);
  void energy_reg(const std::vector<cv::Vec3d>& surface_points,
                  cv::Affine3f inverse_pose);

  void warp(std::vector<Vec3d>& points, std::vector<Vec3d>& normals) const;
  void warp(cuda::Cloud& points) const;

  utils::DualQuaternion<double> DQB(const Vec3f& vertex) const;
  utils::DualQuaternion<double> DQB(const Vec3f& vertex,
                                    const std::vector<double*> epsilon) const;
  utils::DualQuaternion<double> DQB_r(
      const Vec3d& vertex, const float weights[KNN_NEIGHBOURS],
      const unsigned long knn_indices_[KNN_NEIGHBOURS])
      const;                                          // [Minhui 2018/1/28]
  void update_nodes(std::vector<double*>& epsilons);  // [Minhui 2018/1/30]
  void update_nodes(const double* epsilon);

  void getWeightsAndUpdateKNN(const Vec3f& vertex,
                              float weights[KNN_NEIGHBOURS]) const;

  float weighting(float squared_dist, float weight) const;
  void KNN(Vec3f point) const;

  void clear();

  const std::vector<deformation_node>* getNodes() const;
  std::vector<deformation_node>* getNodes();
  const cv::Mat getNodesAsMat() const;
  void setWarpToLive(const Affine3f& pose);
  std::vector<float>* getDistSquared() const;
  std::vector<size_t>* getRetIndex() const;
  // void updateTransformations(double[KNN_NEIGHBOURS][6]); // [Minhui
  // 2018/1/29]

  cv::Vec2d project(const double& x, const double& y, const double& z,
                    const Intr& intr) const;
  cv::Vec3d reproject(const double& u, const double& v, const double& depth,
                      const Intr& intr) const;

  std::vector<cv::Vec3d> live_vertices_;

 private:
  std::vector<deformation_node>* nodes_;
  kd_tree_t* index_;
  Affine3f warp_to_live_;
  void buildKDTree();
};
}
#endif  // KFUSION_WARP_FIELD_HPP
