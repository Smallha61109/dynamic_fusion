#include <cmath>
#include <cstdio>
#include <iostream>
#include <kfusion/warp_field.hpp>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <kfusion/optimisation.hpp>

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    kfusion::WarpField warpField;
    std::vector<cv::Vec3f> warp_init;
    std::vector<cv::Vec3f> warp_normals;
    for(int i=0; i < KNN_NEIGHBOURS; i++)
        warp_normals.emplace_back(cv::Vec3f(0,0,1));

    warp_init.emplace_back(cv::Vec3f(1,1,1));
    warp_init.emplace_back(cv::Vec3f(1,1,-1));
    warp_init.emplace_back(cv::Vec3f(1,-1,1));
    warp_init.emplace_back(cv::Vec3f(1,-1,-1));
    warp_init.emplace_back(cv::Vec3f(-1,1,1));
    warp_init.emplace_back(cv::Vec3f(-1,1,-1));
    warp_init.emplace_back(cv::Vec3f(-1,-1,1));
    warp_init.emplace_back(cv::Vec3f(-1,-1,-1));

    warpField.init(warp_init, warp_normals);

    std::vector<cv::Vec3f> canonical_vertices;
    canonical_vertices.emplace_back(cv::Vec3f(-2,-2,-2));
    canonical_vertices.emplace_back(cv::Vec3f(0,0,0));
    canonical_vertices.emplace_back(cv::Vec3f(2,2,2));
    canonical_vertices.emplace_back(cv::Vec3f(3,3,3));
    canonical_vertices.emplace_back(cv::Vec3f(4,4,4));

    std::vector<cv::Vec3f> canonical_normals;
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));

    std::vector<cv::Vec3f> live_vertices;
    live_vertices.emplace_back(cv::Vec3f(-1.95f,-1.95f,-1.95f));
    live_vertices.emplace_back(cv::Vec3f(0.05,0.05,0.05));
    live_vertices.emplace_back(cv::Vec3f(2.05,2.05,2.05));
    live_vertices.emplace_back(cv::Vec3f(3.05,3.05,3.05));
    live_vertices.emplace_back(cv::Vec3f(4.05,4.05,4.05));

    std::vector<cv::Vec3f> live_normals;
    live_normals.emplace_back(cv::Vec3f(0,0,1));
    live_normals.emplace_back(cv::Vec3f(0,0,1));
    live_normals.emplace_back(cv::Vec3f(0,0,1));
    live_normals.emplace_back(cv::Vec3f(0,0,1));
    live_normals.emplace_back(cv::Vec3f(0,0,1));

    warpField.energy_data(canonical_vertices, canonical_normals,live_vertices, live_normals);
    return 0;
}
