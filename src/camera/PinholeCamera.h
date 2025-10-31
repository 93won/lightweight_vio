/**
 * @file      PinholeCamera.h
 * @brief     Pinhole camera model with Brown-Conrady distortion
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-10-31
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef LIGHTWEIGHT_VIO_PINHOLE_CAMERA_H
#define LIGHTWEIGHT_VIO_PINHOLE_CAMERA_H

#include "camera/Camera.h"
#include "util/Config.h"  // For CameraModel enum
#include <Eigen/Dense>

namespace lightweight_vio {

/**
 * @brief Pinhole camera with Brown-Conrady distortion model
 * 
 * Distortion model: radial (k1, k2) + tangential (p1, p2)
 * Used for standard cameras (EuRoC, TUM RGBD, etc.)
 */
class PinholeCamera : public Camera {
public:
    /**
     * @brief Constructor
     * @param fx Focal length in x
     * @param fy Focal length in y
     * @param cx Principal point x
     * @param cy Principal point y
     * @param k1 Radial distortion coefficient 1
     * @param k2 Radial distortion coefficient 2
     * @param p1 Tangential distortion coefficient 1
     * @param p2 Tangential distortion coefficient 2
     * @param image_width Image width
     * @param image_height Image height
     */
    PinholeCamera(double fx, double fy, double cx, double cy,
                  double k1, double k2, double p1, double p2,
                  int image_width, int image_height);
    
    ~PinholeCamera() override = default;
    
    cv::Point2f project(const Eigen::Vector3f& point_3d) const override;
    Eigen::Vector3f unproject(const cv::Point2f& pixel) const override;
    cv::Point2f undistort(const cv::Point2f& distorted_pixel) const override;
    CameraModel get_model() const override { return CameraModel::PINHOLE; }

private:
    double m_k1, m_k2;  ///< Radial distortion coefficients
    double m_p1, m_p2;  ///< Tangential distortion coefficients
};

} // namespace lightweight_vio

#endif // LIGHTWEIGHT_VIO_PINHOLE_CAMERA_H
