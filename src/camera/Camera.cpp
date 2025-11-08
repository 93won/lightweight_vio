/**
 * @file      Camera.cpp
 * @brief     Implementation of abstract Camera class
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-09
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Camera.h"

namespace lightweight_vio {

Camera::Camera(double fx, double fy, double cx, double cy, 
               const std::vector<double>& distortion_coeffs)
    : m_fx(fx)
    , m_fy(fy)
    , m_cx(cx)
    , m_cy(cy)
    , m_distortion_coeffs(distortion_coeffs)
{
}

std::vector<cv::Point2f> Camera::undistort_points(
    const std::vector<cv::Point2f>& distorted_points) const 
{
    std::vector<cv::Point2f> undistorted_points;
    undistorted_points.reserve(distorted_points.size());
    
    for (const auto& pt : distorted_points) {
        undistorted_points.push_back(undistort_point(pt));
    }
    
    return undistorted_points;
}

cv::Point2f Camera::project_normalized_to_pixel(const Eigen::Vector2f& normalized) const {
    cv::Point2f pixel;
    pixel.x = static_cast<float>(normalized.x() * m_fx + m_cx);
    pixel.y = static_cast<float>(normalized.y() * m_fy + m_cy);
    return pixel;
}

Eigen::Vector2f Camera::compute_normalized(const cv::Point2f& undistorted_pixel) const {
    // Normalize: (u - cx) / fx, (v - cy) / fy
    // This forms the unnormalized bearing vector [x_n, y_n, 1.0]
    return Eigen::Vector2f(
        (undistorted_pixel.x - m_cx) / m_fx,
        (undistorted_pixel.y - m_cy) / m_fy
    );
}

cv::Mat Camera::get_camera_matrix() const {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = m_fx;
    K.at<double>(1, 1) = m_fy;
    K.at<double>(0, 2) = m_cx;
    K.at<double>(1, 2) = m_cy;
    return K;
}

cv::Mat Camera::get_distortion_mat() const {
    cv::Mat D(m_distortion_coeffs.size(), 1, CV_64F);
    for (size_t i = 0; i < m_distortion_coeffs.size(); ++i) {
        D.at<double>(i, 0) = m_distortion_coeffs[i];
    }
    return D;
}

} // namespace lightweight_vio
