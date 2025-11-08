/**
 * @file      Fisheye.cpp
 * @brief     Implementation of Fisheye camera model
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-09
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Fisheye.h"

namespace lightweight_vio {

Fisheye::Fisheye(double fx, double fy, double cx, double cy, 
                 const std::vector<double>& distortion_coeffs)
    : Camera(fx, fy, cx, cy, distortion_coeffs)
{
}

cv::Point2f Fisheye::undistort_point(const cv::Point2f& distorted_point) const {
    // Get camera matrix and distortion coefficients
    cv::Mat K = get_camera_matrix();
    cv::Mat D = get_distortion_mat();
    
    // Use OpenCV's fisheye::undistortPoints
    std::vector<cv::Point2f> distorted_pts = {distorted_point};
    std::vector<cv::Point2f> normalized_pts;
    
    cv::fisheye::undistortPoints(distorted_pts, normalized_pts, K, D);
    
    // Project back to pixel coordinates
    cv::Point2f undistorted_pixel;
    undistorted_pixel.x = static_cast<float>(normalized_pts[0].x * m_fx + m_cx);
    undistorted_pixel.y = static_cast<float>(normalized_pts[0].y * m_fy + m_cy);
    
    return undistorted_pixel;
}

std::vector<cv::Point2f> Fisheye::undistort_points(
    const std::vector<cv::Point2f>& distorted_points) const 
{
    if (distorted_points.empty()) {
        return {};
    }
    
    // Get camera matrix and distortion coefficients
    cv::Mat K = get_camera_matrix();
    cv::Mat D = get_distortion_mat();
    
    // Batch undistortion to normalized coordinates
    std::vector<cv::Point2f> normalized_pts;
    cv::fisheye::undistortPoints(distorted_points, normalized_pts, K, D);
    
    // Convert normalized coordinates back to pixel coordinates
    std::vector<cv::Point2f> undistorted_pixels;
    undistorted_pixels.reserve(normalized_pts.size());
    
    for (const auto& norm_pt : normalized_pts) {
        cv::Point2f pixel;
        pixel.x = static_cast<float>(norm_pt.x * m_fx + m_cx);
        pixel.y = static_cast<float>(norm_pt.y * m_fy + m_cy);
        undistorted_pixels.push_back(pixel);
    }
    
    return undistorted_pixels;
}

} // namespace lightweight_vio
