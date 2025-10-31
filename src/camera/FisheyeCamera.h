/**
 * @file      FisheyeCamera.h
 * @brief     Fisheye camera model with Kannala-Brandt distortion
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-10-31
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef LIGHTWEIGHT_VIO_FISHEYE_CAMERA_H
#define LIGHTWEIGHT_VIO_FISHEYE_CAMERA_H

#include "camera/Camera.h"
#include "util/Config.h"  // For CameraModel enum
#include <Eigen/Dense>

namespace lightweight_vio {

/**
 * @brief Fisheye camera with Kannala-Brandt distortion model
 * 
 * Distortion model: equidistant projection with polynomial distortion
 * Used for wide-angle fisheye cameras (TUM VI, etc.)
 */
class FisheyeCamera : public Camera {
public:
    /**
     * @brief Constructor
     * @param fx Focal length in x
     * @param fy Focal length in y
     * @param cx Principal point x
     * @param cy Principal point y
     * @param k1 Distortion coefficient 1
     * @param k2 Distortion coefficient 2
     * @param k3 Distortion coefficient 3
     * @param k4 Distortion coefficient 4
     * @param image_width Image width
     * @param image_height Image height
     */
    FisheyeCamera(double fx, double fy, double cx, double cy,
                  double k1, double k2, double k3, double k4,
                  int image_width, int image_height);
    
    ~FisheyeCamera() override = default;
    
    cv::Point2f project(const Eigen::Vector3f& point_3d) const override;
    Eigen::Vector3f unproject(const cv::Point2f& pixel) const override;
    cv::Point2f undistort(const cv::Point2f& distorted_pixel) const override;
    CameraModel get_model() const override { return CameraModel::FISHEYE; }

private:
    double m_k1, m_k2, m_k3, m_k4;  ///< Kannala-Brandt distortion coefficients
};

} // namespace lightweight_vio

#endif // LIGHTWEIGHT_VIO_FISHEYE_CAMERA_H
