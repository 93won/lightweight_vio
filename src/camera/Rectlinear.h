/**
 * @file      Rectlinear.h
 * @brief     Rectlinear (pinhole) camera model with radial-tangential distortion
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-09
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef LIGHTWEIGHT_VIO_RECTLINEAR_H
#define LIGHTWEIGHT_VIO_RECTLINEAR_H

#include "Camera.h"

namespace lightweight_vio {

/**
 * @brief Rectlinear (Pinhole) camera model
 * 
 * Standard pinhole camera model with radial-tangential distortion.
 * Uses OpenCV's undistortPoints for distortion removal.
 * 
 * Distortion model (Brown-Conrady):
 * - k1, k2, k3: Radial distortion coefficients
 * - p1, p2: Tangential distortion coefficients
 * 
 * Distortion coefficients order: [k1, k2, p1, p2, k3]
 */
class Rectlinear : public Camera {
public:
    /**
     * @brief Constructor for rectlinear (pinhole) camera
     * @param fx Focal length in x direction
     * @param fy Focal length in y direction
     * @param cx Principal point x coordinate
     * @param cy Principal point y coordinate
     * @param distortion_coeffs Distortion coefficients [k1, k2, p1, p2, k3, ...]
     */
    Rectlinear(double fx, double fy, double cx, double cy, 
               const std::vector<double>& distortion_coeffs);
    
    ~Rectlinear() override = default;
    
    /**
     * @brief Undistort a point using pinhole model
     * @param distorted_point Point in distorted pixel coordinates
     * @return Point in undistorted pixel coordinates
     */
    cv::Point2f undistort_point(const cv::Point2f& distorted_point) const override;
    
    /**
     * @brief Distort a point using pinhole model
     * @param undistorted_point Point in undistorted pixel coordinates
     * @return Point in distorted pixel coordinates
     * @note Uses cv::projectPoints with radial-tangential distortion
     */
    cv::Point2f distort_point(const cv::Point2f& undistorted_point) const override;
    
    /**
     * @brief Batch undistortion optimized for pinhole model
     * @param distorted_points Vector of distorted points
     * @return Vector of undistorted points
     */
    std::vector<cv::Point2f> undistort_points(
        const std::vector<cv::Point2f>& distorted_points) const override;
};

} // namespace lightweight_vio

#endif // LIGHTWEIGHT_VIO_RECTLINEAR_H
