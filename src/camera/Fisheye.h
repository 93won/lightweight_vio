/**
 * @file      Fisheye.h
 * @brief     Fisheye camera model with Kannala-Brandt distortion
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-09
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef LIGHTWEIGHT_VIO_FISHEYE_H
#define LIGHTWEIGHT_VIO_FISHEYE_H

#include "Camera.h"

namespace lightweight_vio {

/**
 * @brief Fisheye camera model
 * 
 * Wide-angle fisheye camera model using the equidistant projection model.
 * Uses OpenCV's fisheye module for distortion removal.
 * 
 * Distortion model (Kannala-Brandt):
 * The model uses equidistant projection:
 * θ = arctan(r/z) where r = sqrt(x^2 + y^2)
 * ρ(θ) = k1*θ + k2*θ^3 + k3*θ^5 + k4*θ^7
 * 
 * Distortion coefficients order: [k1, k2, k3, k4]
 */
class Fisheye : public Camera {
public:
    /**
     * @brief Constructor for fisheye camera
     * @param fx Focal length in x direction
     * @param fy Focal length in y direction
     * @param cx Principal point x coordinate
     * @param cy Principal point y coordinate
     * @param distortion_coeffs Distortion coefficients [k1, k2, k3, k4]
     */
    Fisheye(double fx, double fy, double cx, double cy, 
            const std::vector<double>& distortion_coeffs);
    
    ~Fisheye() override = default;
    
    /**
     * @brief Undistort a point using fisheye model
     * @param distorted_point Point in distorted pixel coordinates
     * @return Point in undistorted pixel coordinates
     */
    cv::Point2f undistort_point(const cv::Point2f& distorted_point) const override;
    
    /**
     * @brief Distort a point using fisheye model
     * @param undistorted_point Point in undistorted pixel coordinates
     * @return Point in distorted pixel coordinates
     * @note Uses cv::fisheye::distortPoints with Kannala-Brandt model
     */
    cv::Point2f distort_point(const cv::Point2f& undistorted_point) const override;
    
    /**
     * @brief Batch undistortion optimized for fisheye model
     * @param distorted_points Vector of distorted points
     * @return Vector of undistorted points
     */
    std::vector<cv::Point2f> undistort_points(
        const std::vector<cv::Point2f>& distorted_points) const override;
};

} // namespace lightweight_vio

#endif // LIGHTWEIGHT_VIO_FISHEYE_H
