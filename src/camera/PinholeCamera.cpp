/**
 * @file      PinholeCamera.cpp
 * @brief     Pinhole camera model implementation
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-10-31
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "camera/PinholeCamera.h"
#include <cmath>

namespace lightweight_vio {

PinholeCamera::PinholeCamera(double fx, double fy, double cx, double cy,
                             double k1, double k2, double p1, double p2,
                             int image_width, int image_height)
    : Camera(fx, fy, cx, cy, image_width, image_height)
    , m_k1(k1), m_k2(k2), m_p1(p1), m_p2(p2)
{
}

cv::Point2f PinholeCamera::project(const Eigen::Vector3f& point_3d) const {
    // Use OpenCV's projectPoints for forward projection
    std::vector<cv::Point3f> object_points = {cv::Point3f(point_3d.x(), point_3d.y(), point_3d.z())};
    std::vector<cv::Point2f> image_points;
    
    cv::Mat K = (cv::Mat_<double>(3, 3) << m_fx, 0, m_cx, 0, m_fy, m_cy, 0, 0, 1);
    cv::Mat D = (cv::Mat_<double>(4, 1) << m_k1, m_k2, m_p1, m_p2);
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    
    cv::projectPoints(object_points, rvec, tvec, K, D, image_points);
    
    return image_points[0];
}

Eigen::Vector3f PinholeCamera::unproject(const cv::Point2f& pixel) const {
    // Use OpenCV's undistortPoints to get normalized coordinates
    std::vector<cv::Point2f> distorted_points = {pixel};
    std::vector<cv::Point2f> normalized_points;
    
    // Camera matrix and distortion coefficients
    cv::Mat K = (cv::Mat_<double>(3, 3) << m_fx, 0, m_cx, 0, m_fy, m_cy, 0, 0, 1);
    cv::Mat D = (cv::Mat_<double>(4, 1) << m_k1, m_k2, m_p1, m_p2);
    
    // undistortPoints gives us normalized coordinates (undistorted and unprojected)
    cv::undistortPoints(distorted_points, normalized_points, K, D);
    
    // Return normalized ray (z=1)
    return Eigen::Vector3f(normalized_points[0].x, normalized_points[0].y, 1.0f).normalized();
}

cv::Point2f PinholeCamera::undistort(const cv::Point2f& distorted_pixel) const {
    // Use OpenCV's undistortPoints for accurate undistortion
    std::vector<cv::Point2f> distorted_points = {distorted_pixel};
    std::vector<cv::Point2f> undistorted_points;
    
    // Camera matrix and distortion coefficients
    cv::Mat K = (cv::Mat_<double>(3, 3) << m_fx, 0, m_cx, 0, m_fy, m_cy, 0, 0, 1);
    cv::Mat D = (cv::Mat_<double>(4, 1) << m_k1, m_k2, m_p1, m_p2);
    
    // OpenCV's undistortPoints removes distortion AND normalizes, so we need to reproject
    cv::undistortPoints(distorted_points, undistorted_points, K, D, cv::noArray(), K);
    
    return undistorted_points[0];
}

} // namespace lightweight_vio
