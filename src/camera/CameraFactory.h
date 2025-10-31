/**
 * @file      CameraFactory.h
 * @brief     Factory for creating camera instances
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-10-31
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef LIGHTWEIGHT_VIO_CAMERA_FACTORY_H
#define LIGHTWEIGHT_VIO_CAMERA_FACTORY_H

#include "util/Config.h"  // For CameraModel enum
#include "camera/Camera.h"
#include "camera/PinholeCamera.h"
#include "camera/FisheyeCamera.h"
#include <memory>
#include <string>

namespace lightweight_vio {

/**
 * @brief Factory class for creating camera instances
 */
class CameraFactory {
public:
    /**
     * @brief Create camera from model type and parameters
     * @param model Camera model (PINHOLE or FISHEYE)
     * @param fx Focal length x
     * @param fy Focal length y
     * @param cx Principal point x
     * @param cy Principal point y
     * @param distortion_coeffs Distortion coefficients (flexible size for different models)
     * @param image_width Image width
     * @param image_height Image height
     * @return Shared pointer to created camera
     * 
     * @note Pinhole uses Brown-Conrady model: [k1, k2, p1, p2] (min 4 coeffs)
     * @note Fisheye uses Kannala-Brandt model: [k1, k2, k3, k4] (min 4 coeffs)
     * @note Future models (e.g., Double Sphere) can use different coefficient counts
     */
    static std::shared_ptr<Camera> create(CameraModel model,
                                         double fx, double fy, double cx, double cy,
                                         const std::vector<double>& distortion_coeffs,
                                         int image_width, int image_height) {
        // Ensure we have at least 4 coefficients (pad with zeros if needed)
        double k1 = distortion_coeffs.size() > 0 ? distortion_coeffs[0] : 0.0;
        double k2 = distortion_coeffs.size() > 1 ? distortion_coeffs[1] : 0.0;
        double k3 = distortion_coeffs.size() > 2 ? distortion_coeffs[2] : 0.0;
        double k4 = distortion_coeffs.size() > 3 ? distortion_coeffs[3] : 0.0;
        
        switch (model) {
            case CameraModel::PINHOLE:
                // For pinhole: k1, k2 are radial, k3=p1, k4=p2 (tangential)
                return std::make_shared<PinholeCamera>(fx, fy, cx, cy, k1, k2, k3, k4, image_width, image_height);
            
            case CameraModel::FISHEYE:
                // For fisheye: k1, k2, k3, k4 are Kannala-Brandt coefficients
                return std::make_shared<FisheyeCamera>(fx, fy, cx, cy, k1, k2, k3, k4, image_width, image_height);
            
            default:
                throw std::runtime_error("Unknown camera model");
        }
    }
    
    /**
     * @brief Create camera from model string ("pinhole" or "fisheye")
     */
    static std::shared_ptr<Camera> create(const std::string& model_str,
                                         double fx, double fy, double cx, double cy,
                                         const std::vector<double>& distortion_coeffs,
                                         int image_width, int image_height) {
        CameraModel model;
        if (model_str == "pinhole") {
            model = CameraModel::PINHOLE;
        } else if (model_str == "fisheye") {
            model = CameraModel::FISHEYE;
        } else {
            throw std::runtime_error("Unknown camera model string: " + model_str);
        }
        
        return create(model, fx, fy, cx, cy, distortion_coeffs, image_width, image_height);
    }
};

} // namespace lightweight_vio

#endif // LIGHTWEIGHT_VIO_CAMERA_FACTORY_H
