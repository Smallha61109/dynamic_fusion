#include "precomp.hpp"
#include "internal.hpp"
#include <tgmath.h>
#include <dual_quaternion.hpp>
#include <nanoflann.hpp>
#include <quaternion.hpp>
#include <knn_point_cloud.hpp>
#include <kfusion/warp_field.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz/vizcore.hpp>

using namespace std;
using namespace kfusion;
using namespace kfusion::cuda;

static inline float deg2rad (float alpha) { return alpha * 0.017453293f; }

/**
 * \brief
 * \return
 */
kfusion::KinFuParams kfusion::KinFuParams::default_params_dynamicfusion()
{
    const int iters[] = {10, 5, 4, 0};
    const int levels = sizeof(iters)/sizeof(iters[0]);

    KinFuParams p;

    p.cols = 640;  //pixels
    p.rows = 480;  //pixels
    p.intr = Intr(570.342f, 570.342f, 320.f, 240.f);

    p.volume_dims = Vec3i::all(256);  //number of voxels
    p.volume_size = Vec3f::all(1.f);  //meters
    p.volume_pose = Affine3f().translate(Vec3f(-p.volume_size[0]/2, -p.volume_size[1]/2, 0.5f));

    p.bilateral_sigma_depth = 0.04f;  //meter
    p.bilateral_sigma_spatial = 4.5; //pixels
    p.bilateral_kernel_size = 7;     //pixels

    p.icp_truncate_depth_dist = 0.f;        //meters, disabled
    p.icp_dist_thres = 0.1f;                //meters
    p.icp_angle_thres = deg2rad(30.f); //radians
    p.icp_iter_num.assign(iters, iters + levels);

    p.tsdf_min_camera_movement = 0.f; //meters, disabled
    p.tsdf_trunc_dist = 0.04f; //meters;
    p.tsdf_max_weight = 64;   //frames

    p.raycast_step_factor = 0.75f;  //in voxel sizes
    p.gradient_delta_factor = 0.5f; //in voxel sizes

    //p.light_pose = p.volume_pose.translation()/4; //meters
    p.light_pose = Vec3f::all(0.f); //meters

    return p;
}

/**
 * \brief
 * \return
 */
kfusion::KinFuParams kfusion::KinFuParams::default_params()
{
    const int iters[] = {10, 5, 4, 0};
    const int levels = sizeof(iters)/sizeof(iters[0]);

    KinFuParams p;
    p.cols = 640;  //pixels
    p.rows = 480;  //pixels
    p.intr = Intr(525.f, 525.f, p.cols/2 - 0.5f, p.rows/2 - 0.5f);

    p.volume_dims = Vec3i::all(512);  //number of voxels
    p.volume_size = Vec3f::all(3.f);  //meters
    p.volume_pose = Affine3f().translate(Vec3f(-p.volume_size[0]/2, -p.volume_size[1]/2, 0.5f));

    p.bilateral_sigma_depth = 0.04f;  //meter
    p.bilateral_sigma_spatial = 4.5; //pixels
    p.bilateral_kernel_size = 7;     //pixels

    p.icp_truncate_depth_dist = 0.f;        //meters, disabled
    p.icp_dist_thres = 0.1f;                //meters
    p.icp_angle_thres = deg2rad(30.f); //radians
    p.icp_iter_num.assign(iters, iters + levels);

    p.tsdf_min_camera_movement = 0.f; //meters, disabled
    p.tsdf_trunc_dist = 0.04f; //meters;
    p.tsdf_max_weight = 64;   //frames

    p.raycast_step_factor = 0.75f;  //in voxel sizes
    p.gradient_delta_factor = 0.5f; //in voxel sizes

    //p.light_pose = p.volume_pose.translation()/4; //meters
    p.light_pose = Vec3f::all(0.f); //meters

    return p;
}

/**
 * \brief
 * \param params
 */
kfusion::KinFu::KinFu(const KinFuParams& params) : frame_counter_(0), params_(params)
{
    CV_Assert(params.volume_dims[0] % 32 == 0);

    volume_ = cv::Ptr<cuda::TsdfVolume>(new cuda::TsdfVolume(params_.volume_dims));
    warp_ = cv::Ptr<WarpField>(new WarpField());

    volume_->setTruncDist(params_.tsdf_trunc_dist);
    volume_->setMaxWeight(params_.tsdf_max_weight);
    volume_->setSize(params_.volume_size);
    volume_->setPose(params_.volume_pose);
    volume_->setRaycastStepFactor(params_.raycast_step_factor);
    volume_->setGradientDeltaFactor(params_.gradient_delta_factor);

    icp_ = cv::Ptr<cuda::ProjectiveICP>(new cuda::ProjectiveICP());
    icp_->setDistThreshold(params_.icp_dist_thres);
    icp_->setAngleThreshold(params_.icp_angle_thres);
    icp_->setIterationsNum(params_.icp_iter_num);

    allocate_buffers();
    reset();
}

const kfusion::KinFuParams& kfusion::KinFu::params() const
{ return params_; }

kfusion::KinFuParams& kfusion::KinFu::params()
{ return params_; }

const kfusion::cuda::TsdfVolume& kfusion::KinFu::tsdf() const
{ return *volume_; }

kfusion::cuda::TsdfVolume& kfusion::KinFu::tsdf()
{ return *volume_; }

const kfusion::cuda::ProjectiveICP& kfusion::KinFu::icp() const
{ return *icp_; }

kfusion::cuda::ProjectiveICP& kfusion::KinFu::icp()
{ return *icp_; }

const kfusion::WarpField& kfusion::KinFu::getWarp() const
{ return *warp_; }

kfusion::WarpField& kfusion::KinFu::getWarp()
{ return *warp_; }

void kfusion::KinFu::allocate_buffers()
{
    const size_t LEVELS = cuda::ProjectiveICP::MAX_PYRAMID_LEVELS;

    int cols = params_.cols;
    int rows = params_.rows;

    dists_.create(rows, cols);

    curr_.depth_pyr.resize(LEVELS);
    curr_.normals_pyr.resize(LEVELS);
    first_.normals_pyr.resize(LEVELS);
    first_.depth_pyr.resize(LEVELS);
    prev_.depth_pyr.resize(LEVELS);
    prev_.normals_pyr.resize(LEVELS);
    first_.normals_pyr.resize(LEVELS);

    curr_.points_pyr.resize(LEVELS);
    prev_.points_pyr.resize(LEVELS);
    first_.points_pyr.resize(LEVELS);

    for(int i = 0; i < LEVELS; ++i)
    {
        curr_.depth_pyr[i].create(rows, cols);
        curr_.normals_pyr[i].create(rows, cols);

        prev_.depth_pyr[i].create(rows, cols);
        prev_.normals_pyr[i].create(rows, cols);

        first_.depth_pyr[i].create(rows, cols);
        first_.normals_pyr[i].create(rows, cols);

        curr_.points_pyr[i].create(rows, cols);
        prev_.points_pyr[i].create(rows, cols);
        first_.points_pyr[i].create(rows, cols);

        cols /= 2;
        rows /= 2;
    }

    depths_.create(params_.rows, params_.cols);
    normals_.create(params_.rows, params_.cols);
    points_.create(params_.rows, params_.cols);
}

void kfusion::KinFu::reset()
{
    if (frame_counter_)
        cout << "Reset" << endl;

    frame_counter_ = 0;
    poses_.clear();
    poses_.reserve(30000);
    poses_.push_back(Affine3f::Identity());
    volume_->clear();
    warp_->clear();
}

/**
 * \brief
 * \param time
 * \return
 */
kfusion::Affine3f kfusion::KinFu::getCameraPose (int time) const
{
    if (time > (int)poses_.size () || time < 0)
        time = (int)poses_.size () - 1;
    return poses_[time];
}

bool kfusion::KinFu::operator()(const kfusion::cuda::Depth& depth, const kfusion::cuda::Image& /*image*/)
{
    const KinFuParams& p = params_;
    const int LEVELS = icp_->getUsedLevelsNum();

    cuda::computeDists(depth, dists_, p.intr);
    cuda::depthBilateralFilter(depth, curr_.depth_pyr[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial, p.bilateral_sigma_depth);

    if (p.icp_truncate_depth_dist > 0)
        kfusion::cuda::depthTruncation(curr_.depth_pyr[0], p.icp_truncate_depth_dist);

    for (int i = 1; i < LEVELS; ++i)
        cuda::depthBuildPyramid(curr_.depth_pyr[i-1], curr_.depth_pyr[i], p.bilateral_sigma_depth);

    for (int i = 0; i < LEVELS; ++i)
#if defined USE_DEPTH
        cuda::computeNormalsAndMaskDepth(p.intr(i), curr_.depth_pyr[i], curr_.normals_pyr[i]);
#else
        cuda::computePointNormals(p.intr(i), curr_.depth_pyr[i], curr_.points_pyr[i], curr_.normals_pyr[i]);
#endif

    cuda::waitAllDefaultStream();

    //can't perform more on first frame
    if (frame_counter_ == 0)
    {

        volume_->integrate(dists_, poses_.back(), p.intr);
        volume_->compute_points();
        volume_->compute_normals();

        warp_->init(volume_->get_cloud_host());

        #if defined USE_DEPTH
        curr_.depth_pyr.swap(prev_.depth_pyr);
        curr_.depth_pyr.swap(first_.depth_pyr);
#else
        curr_.points_pyr.swap(prev_.points_pyr);
        curr_.points_pyr.swap(first_.points_pyr);
#endif
        curr_.normals_pyr.swap(prev_.normals_pyr);
        curr_.normals_pyr.swap(first_.normals_pyr);
        return ++frame_counter_, false;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // ICP
    Affine3f affine; // curr -> prev
    {
        //ScopeTime time("icp");
#if defined USE_DEPTH
        bool ok = icp_->estimateTransform(affine, p.intr, curr_.depth_pyr, curr_.normals_pyr, prev_.depth_pyr, prev_.normals_pyr);
#else
        bool ok = icp_->estimateTransform(affine, p.intr, curr_.points_pyr, curr_.normals_pyr, prev_.points_pyr, prev_.normals_pyr);
#endif
		if (!ok)
			return reset(), false;
		else
			std::cout << "ok" << std::endl;
    }

    //affine = Affine3f::Identity();
    poses_.push_back(poses_.back() * affine); // curr -> global
//    auto d = depth;
    auto d = curr_.depth_pyr[0];
    auto pts = curr_.points_pyr[0];
    auto n = curr_.normals_pyr[0];
    dynamicfusion(d, pts, n);
	// volume_->integrate(dists_, poses_.back(), p.intr);


    ///////////////////////////////////////////////////////////////////////////////////////////
    // Ray casting
    {
        //ScopeTime time("ray-cast-all");
#if defined USE_DEPTH
        volume_->raycast(poses_.back(), p.intr, prev_.depth_pyr[0], prev_.normals_pyr[0]);
        for (int i = 1; i < LEVELS; ++i)
            resizeDepthNormals(prev_.depth_pyr[i-1], prev_.normals_pyr[i-1], prev_.depth_pyr[i], prev_.normals_pyr[i]);
#else
        volume_->raycast(poses_.back(), p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]);
        for (int i = 1; i < LEVELS; ++i)
            resizePointsNormals(prev_.points_pyr[i-1], prev_.normals_pyr[i-1], prev_.points_pyr[i], prev_.normals_pyr[i]);
#endif
        cuda::waitAllDefaultStream();
    }

    return ++frame_counter_, true;
}

/**
 * \brief
 * \param image
 * \param flag
 */
void kfusion::KinFu::renderImage(cuda::Image& image, int flag)
{
    const KinFuParams& p = params_;
    image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);

#if defined USE_DEPTH
    #define PASS1 prev_.depth_pyr
#else
    #define PASS1 prev_.points_pyr
#endif

    if (flag < 1 || flag > 3)
        cuda::renderImage(PASS1[0], prev_.normals_pyr[0], params_.intr, params_.light_pose, image);
    else if (flag == 2)
        cuda::renderTangentColors(prev_.normals_pyr[0], image);
    else /* if (flag == 3) */
    {
        DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
        DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

        cuda::renderImage(PASS1[0], prev_.normals_pyr[0], params_.intr, params_.light_pose, i1);
        cuda::renderTangentColors(prev_.normals_pyr[0], i2);

    }
#undef PASS1
}

/**
 * \brief
 * \param image
 * \param flag
 */
void kfusion::KinFu::dynamicfusion(cuda::Depth& depth, cuda::Cloud current_frame, cuda::Normals current_normals)
{
    cuda::Cloud cloud;
    cuda::Normals normals;
    cloud.create(depth.rows(), depth.cols());
    normals.create(depth.rows(), depth.cols());
    auto camera_pose = poses_.back();
    tsdf().raycast(camera_pose, params_.intr, cloud, normals); // [Minhui 2018/1/27](Need to be checked)"cloud" might be the surface of visible canonical model seen from camera_pose 

    cv::Mat cloud_host(depth.rows(), depth.cols(), CV_32FC4);
    cloud.download(cloud_host.ptr<Point>(), cloud_host.step);
    std::vector<Vec3f> warped(cloud_host.rows * cloud_host.cols); // [Minhui 2018/1/27]"cloud_host.rows * cloud_host.cols" always equals to 307200(640*480)
    auto inverse_pose = camera_pose.inv(cv::DECOMP_SVD); // [Minhui 2018/1/27]"camera_pose" is obtained from icp in KinectFusion
    // [Minhui 2018/1/28]Why inverse: transform the corrdinate to the first frame (probably the global coordinate)
    
    //TODO: check here
    for (int i = 0; i < cloud_host.rows; i++)
    {
        for (int j = 0; j < cloud_host.cols; j++) {
            Point point = cloud_host.at<Point>(i, j);
            warped[i * cloud_host.cols + j][0] = point.x; // [Minhui 2018/1/27]First, store the canonical to "warped"
            warped[i * cloud_host.cols + j][1] = point.y;
            warped[i * cloud_host.cols + j][2] = point.z;
            warped[i * cloud_host.cols + j] = inverse_pose * warped[i * cloud_host.cols + j]; //[Minhui 2018/1/27]Second, eliminate the influence of rigid camera motion
        }
    }
    cv::Mat normal_host(cloud_host.rows, cloud_host.cols, CV_32FC4);
    normals.download(normal_host.ptr<Normal>(), normal_host.step);

    std::vector<Vec3f> warped_normals(normal_host.rows * normal_host.cols);
    for (int i = 0; i < normal_host.rows; i++)
    {
        for (int j = 0; j < normal_host.cols; j++) {
            auto point = normal_host.at<Normal>(i, j);
            warped_normals[i * normal_host.cols + j][0] = point.x;
            warped_normals[i * normal_host.cols + j][1] = point.y;
            warped_normals[i * normal_host.cols + j][2] = point.z;
            // [Minhui 2018/1/27]TODO: Warped_normals may also need to be multiplied with inverse_pose
            warped_normals[i * normal_host.cols + j] = inverse_pose * warped[i * cloud_host.cols + j]; // [Minhui 2018/1/28]add
        }
    }
    // std::vector<Vec3f> canonical_visible(warped);
    // FIXME: fix energy regularization and all energy function
    //  FIXME: make up values for debuging.
    cv::Matx33f warp_rot_mat(1, 2, 3, 4, 5, 6, 7, 8, 9);
    cv::Vec3f warp_trans_vec(1, 2, 3);
    cv::Vec3f one_point(8, 8, 8);
    std::vector<cv::Vec3f> surface_points;
    std::vector<cv::Matx33f> warp_rot_all;
    std::vector<cv::Vec3f> warp_trans_all;
    surface_points.push_back(one_point);
    warp_rot_all.push_back(warp_rot_mat);
    warp_trans_all.push_back(warp_trans_vec);
    getWarp().energy_reg(surface_points, inverse_pose, warp_rot_all, warp_trans_all);

    /* [Minhui 2018/1/27] Transform the data structure of "current_frame" and "current_normals" from cuda::Cloud/Normals to std::vector<Vec3f> */
    cv::Mat live_cloud_host(current_frame.rows(), current_frame.cols(), CV_32FC4); // TODO: check "CV_32FC4"
    current_frame.download(live_cloud_host.ptr<Point>(), live_cloud_host.step);
    std::vector<Vec3f> live(live_cloud_host.rows * live_cloud_host.cols);
    for (int i = 0; i < live_cloud_host.rows; i++) {
        for (int j = 0; j < live_cloud_host.cols; j++) {
            Point point = live_cloud_host.at<Point>(i, j);
            live[i * live_cloud_host.cols + j][0] = point.x;
            live[i * live_cloud_host.cols + j][1] = point.y;
            live[i * live_cloud_host.cols + j][2] = point.z;
        }
    }

    cv::Mat live_normal_host(current_normals.rows(), current_normals.cols(), CV_32FC4);
    current_normals.download(live_normal_host.ptr<Normal>(), live_normal_host.step);
    std::vector<Vec3f> live_normals(live_normal_host.rows * live_normal_host.cols);
    for (int i = 0; i < live_normal_host.rows; i++) {
        for (int j = 0; j < live_normal_host.cols; j++) {
            auto point = live_normal_host.at<Normal>(i, j);
            live_normals[i * live_normal_host.cols + j][0] = point.x;
            live_normals[i * live_normal_host.cols + j][1] = point.y;
            live_normals[i * live_normal_host.cols + j][2] = point.z;
        }
    }
    /* (End) */

    // [Original] getWarp().energy_data(warped, warped_normals, warped, warped_normals); //crashes, leave out for now
    // cv::imshow("cloud_host", cloud_host);
    // cv::imshow("live_cloud_host", live_cloud_host);
    // cv::imshow("normal_host", normal_host);
    // cv::imshow("live_normal_host", live_normal_host);
    // cv::waitKey(10);

    std::vector<Vec3d> warped_d(warped.begin(), warped.end());
    std::vector<Vec3d> warped_normals_d(warped_normals.begin(), warped_normals.end());
    std::vector<Vec3d> live_d(live.begin(), live.end());
    std::vector<Vec3d> live_normals_d(live_normals.begin(), live_normals.end());
    std::vector<Vec3d> canonical_visible(warped_d);

    getWarp().energy_data(warped_d, warped_normals_d, live_d, live_normals_d, params_.intr);

    // TODO: getwarp.warp() always run into nan so this is not functioning
    getWarp().warp(warped_d, warped_normals_d);
//    //ScopeTime time("fusion");
    tsdf().surface_fusion(getWarp(), warped_d, canonical_visible, depth, camera_pose, params_.intr);

    cv::Mat depth_cloud(depth.rows(),depth.cols(), CV_16U);
    depth.download(depth_cloud.ptr<void>(), depth_cloud.step);
    cv::Mat display;
    depth_cloud.convertTo(display, CV_8U, 255.0/4000);
    cv::imshow("Depth diff", display);
    volume_->compute_points();
    volume_->compute_normals();
}

/**
 * \brief
 * \param image
 * \param pose
 * \param flag
 */
void kfusion::KinFu::renderImage(cuda::Image& image, const Affine3f& pose, int flag) {
    const KinFuParams &p = params_;
    image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);
    depths_.create(p.rows, p.cols);
    normals_.create(p.rows, p.cols);
    points_.create(p.rows, p.cols);

#if defined USE_DEPTH
#define PASS1 depths_
#else
#define PASS1 points_
#endif

    volume_->raycast(pose, p.intr, PASS1, normals_);

    if (flag < 1 || flag > 3)
        cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, image);
    else if (flag == 2)
        cuda::renderTangentColors(normals_, image);
    else /* if (flag == 3) */
    {
        DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
        DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

        cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, i1);
        cuda::renderTangentColors(normals_, i2);
    }
#undef PASS1
}
