#include <iostream>
#include <kfusion/kinfu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>

using namespace kfusion;

void show_depth(const cv::Mat& depth) {
  cv::Mat display;
  depth.convertTo(display, CV_8U, 255.0 / 4000);
  cv::imshow("Depth", display);
}


struct DynamicFusionApp {

  DynamicFusionApp(std::string dir, bool visualize, const KinFuParams& params)
      : iter(0),
        exit_(false),
        interactive_mode_(false),
        pause_(false),
        directory(true),
        dir_name(dir) {
    kinfu_ = KinFu::Ptr(new KinFu(params));
  }
  bool step(int index) {
    KinFu& dynamic_fusion = *kinfu_;
    image = cv::imread(images[index], CV_LOAD_IMAGE_COLOR);
    depth = cv::imread(depths[index], CV_LOAD_IMAGE_ANYDEPTH);
    depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);

    // SampledScopeTime fps(time_ms);
    // (void)fps;
    has_image = dynamic_fusion(depth_device_);
  }

  bool execute() {
    cv::Mat depth, image;
    double time_ms = 0;
    bool has_image = false;

    cv::glob(dir_name + "/depth", depths);
    cv::glob(dir_name + "/color", images);

    std::sort(depths.begin(), depths.end());
    std::sort(images.begin(), images.end());

    return true;
  }
  bool done(){
    return iter >= depths.size();
  }

  int iter;
  bool has_image;
  bool pause_ /*= false*/;
  bool exit_, interactive_mode_, directory;
  std::string dir_name;
  KinFu::Ptr kinfu_;
  std::vector<cv::String> depths;  // store paths,
  std::vector<cv::String> images;  // store paths,
  cv::Mat image;
  cv::Mat depth;

  cv::Mat view_host_;
  cuda::Image view_device_;
  cuda::Depth depth_device_;
};

class visualize {
 public:
  visualize(const KinFuParams& params, DynamicFusionApp* app):
    app_(app)
  {
    cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true,
                        cv::viz::Color::apricot());
    viz.showWidget("cube", cube, params.volume_pose);
    viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
    viz.registerKeyboardCallback(KeyboardCallback, this);
  }

  static void KeyboardCallback(const cv::viz::KeyboardEvent& event,
                               void* pthis) {
    visualize* v = static_cast<visualize*>(pthis);
    DynamicFusionApp* kinfu = v->app_;

    if (event.action != cv::viz::KeyboardEvent::KEY_DOWN) return;

    if (event.code == 't' || event.code == 'T')
      v->show_warp(*kinfu->kinfu_);

    if (event.code == 'i' || event.code == 'I')
      kinfu->interactive_mode_ = !kinfu->interactive_mode_;
  }

  void show_raycasted(KinFu& kinfu, cv::Mat& view_host,
                      cuda::Image& view_device) {
    const int mode = 3;
    if (app_->interactive_mode_)
      kinfu.renderImage(view_device, viz.getViewerPose(), mode);
    else
      kinfu.renderImage(view_device, mode);

    view_host.create(view_device.rows(), view_device.cols(), CV_8UC4);
    view_device.download(view_host.ptr<void>(), view_host.step);

    static int scene_frame_idx = 1;
    std::string scene_file = "scene_" + std::to_string(scene_frame_idx);
    scene_file = scene_file + ".jpg";
    cv::imshow("Scene", view_host);
    // cv::imwrite(scene_file, view_host_);
    scene_frame_idx++;
  }

  void show_warp(KinFu& kinfu) {
    cv::Mat warp_host = kinfu.getWarp().getNodesAsMat();
    viz1.showWidget("warp_field", cv::viz::WCloud(warp_host));
  }

  bool step(const cv::Mat& image, const cv::Mat& depth) {
    KinFu& dynamic_fusion = *(app_->kinfu_);
    show_depth(depth);
    cv::imshow("Image", image);
    if (app_->has_image) show_raycasted(dynamic_fusion, app_->view_host_, app_->view_device_);
    if (!app_->interactive_mode_) {
      viz.setViewerPose(dynamic_fusion.getCameraPose());
      viz1.setViewerPose(dynamic_fusion.getCameraPose());
    }
    show_warp(dynamic_fusion);
    int key = cv::waitKey(app_->pause_ ? 0 : 1000);
    switch (key) {
      case 't':
      case 'T':
        show_warp(dynamic_fusion);
        break;
      case 'i':
      case 'I':
        app_->interactive_mode_ = !app_->interactive_mode_;
        break;
      case 27:
        app_->exit_ = true;
        break;
      case 32:
        app_->pause_ = !app_->pause_;
        break;
    }
    viz.spinOnce(3, true);
    viz1.spinOnce(3, true);
    return true;
  }
  DynamicFusionApp* app_;

 private:
  cv::viz::Viz3d viz;
  cv::viz::Viz3d viz1;
};

/////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
  int device = 0;
  bool v = false;
  cuda::setDevice(device);
  cuda::printShortCudaDeviceInfo(device);

  if (cuda::checkIfPreFermiGPU(device))
    return std::cout << std::endl
                     << "Kinfu is not supported for pre-Fermi GPU "
                        "architectures, and not built for them by default. "
                        "Exiting..."
                     << std::endl,
           -1;

  if (argc == 3) v = argv[2];
  KinFuParams params = KinFuParams::default_params_dynamicfusion();
  DynamicFusionApp* app = new DynamicFusionApp(argv[1], v, params);
  visualize* visu = NULL;
  if(v)
    visualize* visu = new visualize(params, app);

  // executing
  try {
    app->execute();
    while(!app->done()){
      app->step(app->iter);
      app->iter++;
      if(v)
        visu->step(app->image, app->depth);
    }
  } catch (const std::bad_alloc& /*e*/) {
    std::cout << "Bad alloc" << std::endl;
  } catch (const std::exception& /*e*/) {
    std::cout << "Exception" << std::endl;
  }

  delete app;
  return 0;
}
