/**
 * @file      Camera.h
 * @brief     Abstract base class for camera models (pinhole, fisheye, etc.)
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-09
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef LIGHTWEIGHT_VIO_CAMERA_H
#define LIGHTWEIGHT_VIO_CAMERA_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <vector>
#include <memory>

namespace lightweight_vio {

/**
 * @brief Abstract base class for camera models
 * 
 * Provides interface for different camera projection models (pinhole, fisheye, etc.)
 * and handles intrinsic parameters and distortion coefficients.
 */
class Camera {
public:
    /**
     * @brief Constructor with intrinsic parameters
     * @param fx Focal length in x direction
     * @param fy Focal length in y direction
     * @param cx Principal point x coordinate
     * @param cy Principal point y coordinate
     * @param distortion_coeffs Distortion coefficients (model-specific)
     */
    Camera(double fx, double fy, double cx, double cy, 
           const std::vector<double>& distortion_coeffs);
    
    virtual ~Camera() = default;
    
    /**
     * @brief Undistort a single point from distorted pixel coordinates to undistorted pixel coordinates
     * @param distorted_point Point in distorted pixel coordinates
     * @return Point in undistorted pixel coordinates
     */
    virtual cv::Point2f undistort_point(const cv::Point2f& distorted_point) const = 0;
    
    /**
     * @brief Distort a single point from undistorted pixel coordinates to distorted pixel coordinates
     * @param undistorted_point Point in undistorted pixel coordinates
     * @return Point in distorted pixel coordinates
     * @note Uses iterative Newton-Raphson method to find distorted coordinates
     */
    virtual cv::Point2f distort_point(const cv::Point2f& undistorted_point) const = 0;
    
    /**
     * @brief Batch undistortion for multiple points
     * @param distorted_points Vector of distorted points
     * @return Vector of undistorted points
     */
    virtual std::vector<cv::Point2f> undistort_points(
        const std::vector<cv::Point2f>& distorted_points) const;
    
    /**
     * @brief Batch distortion for multiple points
     * @param undistorted_points Vector of undistorted points
     * @return Vector of distorted points
     */
    virtual std::vector<cv::Point2f> distort_points(
        const std::vector<cv::Point2f>& undistorted_points) const;
    
    /**
     * @brief Project normalized coordinates to pixel coordinates
     * @param normalized Normalized coordinates (x/z, y/z)
     * @return Pixel coordinates
     */
    cv::Point2f project_normalized_to_pixel(const Eigen::Vector2f& normalized) const;
    
    /**
     * @brief Compute normalized camera coordinates from undistorted pixel
     * @param undistorted_pixel Undistorted pixel coordinates
     * @return Normalized coordinates [x_n, y_n] where x_n = (u - cx)/fx, y_n = (v - cy)/fy
     * @note Forms unnormalized bearing vector: [x_n, y_n, 1.0]
     */
    Eigen::Vector2f compute_normalized(const cv::Point2f& undistorted_pixel) const;
    
    // Getters
    double get_fx() const { return m_fx; }
    double get_fy() const { return m_fy; }
    double get_cx() const { return m_cx; }
    double get_cy() const { return m_cy; }
    const std::vector<double>& get_distortion_coeffs() const { return m_distortion_coeffs; }
    
    /**
     * @brief Get camera intrinsic matrix K
     * @return 3x3 camera matrix
     */
    cv::Mat get_camera_matrix() const;
    
    /**
     * @brief Get distortion coefficients as cv::Mat
     * @return Distortion coefficients
     */
    cv::Mat get_distortion_mat() const;
    
protected:
    double m_fx;  ///< Focal length in x direction
    double m_fy;  ///< Focal length in y direction
    double m_cx;  ///< Principal point x coordinate
    double m_cy;  ///< Principal point y coordinate
    std::vector<double> m_distortion_coeffs;  ///< Distortion coefficients
};

} // namespace lightweight_vio

#endif // LIGHTWEIGHT_VIO_CAMERA_H
