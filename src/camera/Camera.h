/**
 * @file      Camera.h
 * @brief     Abstract camera model base class
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-10-31
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef LIGHTWEIGHT_VIO_CAMERA_H
#define LIGHTWEIGHT_VIO_CAMERA_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

namespace lightweight_vio {

// Forward declarations (enums defined in Config.h)
enum class CameraModel;
enum class CameraType;

/**
 * @brief Abstract base class for camera models
 * 
 * Provides interface for projection, unprojection, and distortion operations.
 * Derived classes implement specific distortion models (pinhole, fisheye, etc.)
 */
class Camera {
public:
    /**
     * @brief Constructor
     * @param fx Focal length in x direction
     * @param fy Focal length in y direction
     * @param cx Principal point x coordinate
     * @param cy Principal point y coordinate
     * @param image_width Image width in pixels
     * @param image_height Image height in pixels
     */
    Camera(double fx, double fy, double cx, double cy, int image_width, int image_height)
        : m_fx(fx), m_fy(fy), m_cx(cx), m_cy(cy)
        , m_image_width(image_width), m_image_height(image_height) {}
    
    virtual ~Camera() = default;
    
    /**
     * @brief Project 3D point to 2D pixel coordinates (with distortion)
     * @param point_3d 3D point in camera frame
     * @return 2D pixel coordinates
     */
    virtual cv::Point2f project(const Eigen::Vector3f& point_3d) const = 0;
    
    /**
     * @brief Unproject 2D pixel to 3D ray (removes distortion)
     * @param pixel Distorted pixel coordinates
     * @return Normalized 3D ray direction in camera frame
     */
    virtual Eigen::Vector3f unproject(const cv::Point2f& pixel) const = 0;
    
    /**
     * @brief Remove distortion from pixel coordinates
     * @param distorted_pixel Distorted pixel coordinates
     * @return Undistorted pixel coordinates
     */
    virtual cv::Point2f undistort(const cv::Point2f& distorted_pixel) const = 0;
    
    /**
     * @brief Get camera model type
     */
    virtual CameraModel get_model() const = 0;
    
    /**
     * @brief Check if pixel is within image bounds
     */
    bool is_in_image(const cv::Point2f& pixel, int border = 0) const {
        return pixel.x >= border && pixel.x < m_image_width - border &&
               pixel.y >= border && pixel.y < m_image_height - border;
    }
    
    // Getters
    double get_fx() const { return m_fx; }
    double get_fy() const { return m_fy; }
    double get_cx() const { return m_cx; }
    double get_cy() const { return m_cy; }
    int get_image_width() const { return m_image_width; }
    int get_image_height() const { return m_image_height; }

protected:
    double m_fx, m_fy;  ///< Focal lengths
    double m_cx, m_cy;  ///< Principal point
    int m_image_width, m_image_height;  ///< Image dimensions
};

} // namespace lightweight_vio

#endif // LIGHTWEIGHT_VIO_CAMERA_H
