/**
 * @file      Config.cpp
 * @brief     Implements the singleton class for managing configuration parameters.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-11
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "util/Config.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <spdlog/spdlog.h>

namespace lightweight_vio {

bool Config::load(const std::string& config_file) {
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file: " << config_file << std::endl;
        return false;
    }
    
    // Performance Parameters - Load first to control debug output
    cv::FileNode performance = fs["performance"];
    if (!performance.empty()) {
        auto enable_debug_output_node = performance["enable_debug_output"];
        
        m_enable_debug_output = (bool)(int)enable_debug_output_node;
    }
    
    // Feature Detection Parameters
    cv::FileNode feature_detection = fs["feature_detection"];
    if (!feature_detection.empty()) {
        m_max_features = (int)feature_detection["max_features"];
        m_quality_level = (double)feature_detection["quality_level"];
        m_min_distance = (double)feature_detection["min_distance"];
        
        // Grid-based feature distribution parameters
        if (!feature_detection["grid_cols"].empty()) {
            m_grid_cols = (int)feature_detection["grid_cols"];
        }
        if (!feature_detection["grid_rows"].empty()) {
            m_grid_rows = (int)feature_detection["grid_rows"];
        }
        if (!feature_detection["max_features_per_grid"].empty()) {
            m_max_features_per_grid = (int)feature_detection["max_features_per_grid"];
        }
    }
    
    // Optical Flow Parameters
    cv::FileNode optical_flow = fs["optical_flow"];
    if (!optical_flow.empty()) {
        m_window_size = (int)optical_flow["window_size"];
        m_max_level = (int)optical_flow["max_level"];
        m_max_iterations = (int)optical_flow["max_iterations"];
        m_epsilon = (double)optical_flow["epsilon"];
        m_error_threshold = (double)optical_flow["error_threshold"];
        m_max_movement = (double)optical_flow["max_movement"];
        m_min_eigen_threshold = (double)optical_flow["min_eigen_threshold"];
        m_F_threshold = (double)optical_flow["F_threshold"];
    }
    
    // Stereo Matching Parameters
    cv::FileNode stereo_matching = fs["stereo_matching"];
    if (!stereo_matching.empty()) {
        // Stereo optical flow parameters
        m_stereo_window_size = (int)stereo_matching["window_size"];
        m_stereo_max_level = (int)stereo_matching["max_level"];
        m_stereo_max_iterations = (int)stereo_matching["max_iterations"];
        m_stereo_min_eigen_threshold = (double)stereo_matching["min_eigen_threshold"];
        
        // Stereo matching thresholds
        m_stereo_error_threshold = (double)stereo_matching["error_threshold"];
        m_min_disparity = (double)stereo_matching["min_disparity"];
        m_max_disparity = (double)stereo_matching["max_disparity"];
        m_max_y_difference = (double)stereo_matching["max_y_difference"];
        m_epipolar_threshold = (double)stereo_matching["epipolar_threshold"];
    }
    
    // Global Depth Parameters (used throughout the VIO system)
    cv::FileNode depth = fs["depth"];
    if (!depth.empty()) {
        m_min_depth = (double)depth["min_depth"];
        m_max_depth = (double)depth["max_depth"];
    }
    
 
    
    // Keyframe Parameters
    cv::FileNode keyframe_mgmt = fs["keyframe_management"];
    if (!keyframe_mgmt.empty()) {
        m_grid_coverage_ratio = (double)keyframe_mgmt["grid_coverage_ratio"];
        m_keyframe_window_size = (int)keyframe_mgmt["keyframe_window_size"];
        m_keyframe_time_threshold = (double)keyframe_mgmt["time_threshold"];
        if (m_enable_debug_output) {
            spdlog::info("Loaded keyframe management from config: grid_coverage_ratio={}, window_size={}, time_threshold={}", 
                        m_grid_coverage_ratio, m_keyframe_window_size, m_keyframe_time_threshold);
        }
    } else {
        if (m_enable_debug_output) {
            spdlog::warn("Keyframe management section not found in config, using defaults: grid_coverage_ratio={}, window_size={}, time_threshold={}", 
                        m_grid_coverage_ratio, m_keyframe_window_size, m_keyframe_time_threshold);
        }
    }
    
    // Optimization Parameters (unified for PnP and Sliding Window)
    cv::FileNode optimization = fs["optimization"];
    if (!optimization.empty()) {
        // PnP specific variables
        if (optimization["pnp_max_iterations"].isInt()) {
            m_pnp_max_iterations = (int)optimization["pnp_max_iterations"];
            m_pose_max_iterations = m_pnp_max_iterations; // Legacy compatibility
        } else if (optimization["max_iterations"].isInt()) {
            // Fallback to old config format
            m_pnp_max_iterations = (int)optimization["max_iterations"];
            m_pose_max_iterations = m_pnp_max_iterations; // Legacy compatibility
        }
       
        if (optimization["function_tolerance"].isReal()) {
            m_pnp_function_tolerance = (double)optimization["function_tolerance"];
            m_pose_function_tolerance = m_pnp_function_tolerance; // Legacy compatibility
        }
        if (optimization["gradient_tolerance"].isReal()) {
            m_pnp_gradient_tolerance = (double)optimization["gradient_tolerance"];
            m_pose_gradient_tolerance = m_pnp_gradient_tolerance; // Legacy compatibility
        }
        if (optimization["parameter_tolerance"].isReal()) {
            m_pnp_parameter_tolerance = (double)optimization["parameter_tolerance"];
            m_pose_parameter_tolerance = m_pnp_parameter_tolerance; // Legacy compatibility
        }
        if (optimization["use_robust_kernel"].isInt()) {
            m_pnp_use_robust_kernel = (bool)(int)optimization["use_robust_kernel"];
            m_use_robust_kernel = m_pnp_use_robust_kernel; // Legacy compatibility
        }
        if (optimization["enable_outlier_detection"].isInt()) {
            m_pnp_enable_outlier_detection = (bool)(int)optimization["enable_outlier_detection"];
            m_enable_outlier_detection = m_pnp_enable_outlier_detection; // Legacy compatibility
        }
        if (optimization["outlier_detection_rounds"].isInt()) {
            m_pnp_outlier_detection_rounds = (int)optimization["outlier_detection_rounds"];
            m_outlier_detection_rounds = m_pnp_outlier_detection_rounds; // Legacy compatibility
        }
        
        // Sliding Window specific parameters  
        if (optimization["sliding_window_max_iterations"].isInt()) {
            m_sw_max_iterations = (int)optimization["sliding_window_max_iterations"];
        }
        if (optimization["sliding_window_max_observation_weight"].isReal()) {
            m_sw_max_observation_weight = (double)optimization["sliding_window_max_observation_weight"];
        }
        
        // Use common parameters for sliding window (function_tolerance, gradient_tolerance, parameter_tolerance)
        m_sw_function_tolerance = m_pnp_function_tolerance;
        m_sw_gradient_tolerance = m_pnp_gradient_tolerance;
        m_sw_parameter_tolerance = m_pnp_parameter_tolerance;
        m_sw_use_robust_kernel = m_pnp_use_robust_kernel;
        
        // Remove logging-related parameters as they're not needed
    }
    
    // Camera Parameters
    cv::FileNode camera = fs["camera"];
    if (!camera.empty()) {
        // Load camera model (pinhole or fisheye)
        if (!camera["model"].empty()) {
            std::string model_str = (std::string)camera["model"];
            if (model_str == "fisheye") {
                m_camera_model = CameraModel::FISHEYE;
            } else {
                m_camera_model = CameraModel::PINHOLE;
            }
        }
        
        // Read camera type from YAML and set enum
        if (camera["type"].isString()) {
            std::string type_str = (std::string)camera["type"];
            if (type_str == "rgbd") {
                m_camera_type = CameraType::RGBD;
                spdlog::info("[CONFIG] Camera type set to RGBD");
            } else if (type_str == "stereo") {
                m_camera_type = CameraType::STEREO;
                spdlog::info("[CONFIG] Camera type set to STEREO");
            } else if (type_str == "monocular") {
                m_camera_type = CameraType::MONOCULAR;
                spdlog::info("[CONFIG] Camera type set to MONOCULAR");
            }
        }
        
        m_image_width = (int)camera["image_width"];
        m_image_height = (int)camera["image_height"];
        m_border_size = (int)camera["border_size"];
        
        
        // Load camera intrinsics based on camera type
        if (m_camera_type == CameraType::RGBD) {
            // For RGBD, use rgb_intrinsics and rgb_distortion
            cv::FileNode rgb_intrinsics = camera["rgb_intrinsics"];
            cv::FileNode rgb_distortion = camera["rgb_distortion"];
            
            if (!rgb_intrinsics.empty() && rgb_intrinsics.size() == 4) {
                m_left_camera_matrix = (cv::Mat_<double>(3, 3) << 
                    (double)rgb_intrinsics[0], 0, (double)rgb_intrinsics[2],
                    0, (double)rgb_intrinsics[1], (double)rgb_intrinsics[3],
                    0, 0, 1);
            }
            
            if (!rgb_distortion.empty() && rgb_distortion.size() == 4) {
                m_left_dist_coeffs = (cv::Mat_<double>(1, 4) << 
                    (double)rgb_distortion[0], (double)rgb_distortion[1],
                    (double)rgb_distortion[2], (double)rgb_distortion[3]);
            }
        } else {
            // For stereo, use left_intrinsics and right_intrinsics
            cv::FileNode left_intrinsics = camera["left_intrinsics"];
            cv::FileNode right_intrinsics = camera["right_intrinsics"];
            cv::FileNode left_distortion = camera["left_distortion"];
            cv::FileNode right_distortion = camera["right_distortion"];
            
            if (!left_intrinsics.empty() && left_intrinsics.size() == 4) {
                m_left_camera_matrix = (cv::Mat_<double>(3, 3) << 
                    (double)left_intrinsics[0], 0, (double)left_intrinsics[2],
                    0, (double)left_intrinsics[1], (double)left_intrinsics[3],
                    0, 0, 1);
            }
            
            if (!right_intrinsics.empty() && right_intrinsics.size() == 4) {
                m_right_camera_matrix = (cv::Mat_<double>(3, 3) << 
                    (double)right_intrinsics[0], 0, (double)right_intrinsics[2],
                    0, (double)right_intrinsics[1], (double)right_intrinsics[3],
                    0, 0, 1);
            }
            
            if (!left_distortion.empty() && left_distortion.size() == 4) {
                m_left_dist_coeffs = (cv::Mat_<double>(1, 4) << 
                    (double)left_distortion[0], (double)left_distortion[1],
                    (double)left_distortion[2], (double)left_distortion[3]);
            }
            
            if (!right_distortion.empty() && right_distortion.size() == 4) {
                m_right_dist_coeffs = (cv::Mat_<double>(1, 4) << 
                    (double)right_distortion[0], (double)right_distortion[1],
                    (double)right_distortion[2], (double)right_distortion[3]);
            }
        }
        
        // Load extrinsics (T_BC - camera to body transform)
        cv::FileNode left_T_BC = m_camera_type == CameraType::RGBD ? camera["rgb_T_BC"] : camera["left_T_BC"];
        cv::FileNode right_T_BC = camera["right_T_BC"];
        
        // For RGBD and MONOCULAR, only left_T_BC is needed
        if (m_camera_type == CameraType::RGBD || m_camera_type == CameraType::MONOCULAR) {
            if (!left_T_BC.empty() && left_T_BC.size() == 16) {
                m_T_left_BC = (cv::Mat_<double>(4, 4) << 
                    (double)left_T_BC[0], (double)left_T_BC[1], (double)left_T_BC[2], (double)left_T_BC[3],
                    (double)left_T_BC[4], (double)left_T_BC[5], (double)left_T_BC[6], (double)left_T_BC[7],
                    (double)left_T_BC[8], (double)left_T_BC[9], (double)left_T_BC[10], (double)left_T_BC[11],
                    (double)left_T_BC[12], (double)left_T_BC[13], (double)left_T_BC[14], (double)left_T_BC[15]);
            }
        } else {
            // For STEREO, both left and right T_BC are needed
            if (!left_T_BC.empty() && !right_T_BC.empty() && 
                left_T_BC.size() == 16 && right_T_BC.size() == 16) {
                
                cv::Mat T_BC_left = (cv::Mat_<double>(4, 4) << 
                    (double)left_T_BC[0], (double)left_T_BC[1], (double)left_T_BC[2], (double)left_T_BC[3],
                    (double)left_T_BC[4], (double)left_T_BC[5], (double)left_T_BC[6], (double)left_T_BC[7],
                    (double)left_T_BC[8], (double)left_T_BC[9], (double)left_T_BC[10], (double)left_T_BC[11],
                    (double)left_T_BC[12], (double)left_T_BC[13], (double)left_T_BC[14], (double)left_T_BC[15]);
                    
                cv::Mat T_BC_right = (cv::Mat_<double>(4, 4) << 
                    (double)right_T_BC[0], (double)right_T_BC[1], (double)right_T_BC[2], (double)right_T_BC[3],
                    (double)right_T_BC[4], (double)right_T_BC[5], (double)right_T_BC[6], (double)right_T_BC[7],
                    (double)right_T_BC[8], (double)right_T_BC[9], (double)right_T_BC[10], (double)right_T_BC[11],
                    (double)right_T_BC[12], (double)right_T_BC[13], (double)right_T_BC[14], (double)right_T_BC[15]);
                
                // Store individual T_BC matrices (camera to body transforms)
                m_T_left_BC = T_BC_left.clone();
                m_T_right_BC = T_BC_right.clone();
                
                // Compute stereo baseline transform: T_left_right = T_BC_right.inv() * T_BC_left
                // This gives left-to-right camera transformation for stereo triangulation
                m_T_left_right = T_BC_right.inv() * T_BC_left;
            }
        }
        
        // Debug output for camera parameters
        if (m_enable_debug_output) {
            spdlog::info("[CONFIG] Camera parameters loaded:");
            spdlog::info("  - Camera model: {}", (m_camera_model == CameraModel::FISHEYE) ? "fisheye" : "pinhole");
            spdlog::info("  - Image size: {}x{}", m_image_width, m_image_height);
            spdlog::info("  - Border size: {}", m_border_size);
        }
    }
    
    // System Mode Parameters
    cv::FileNode system_mode = fs["system_mode"];
    if (!system_mode.empty()) {
        m_system_mode = (std::string)system_mode["mode"];
        if (m_enable_debug_output) {
            spdlog::info("[CONFIG] System mode: {}", m_system_mode);
        }
    }
    
    // Viewer Parameters
    cv::FileNode viewer = fs["viewer"];
    if (!viewer.empty()) {
        m_viewer_enable = (bool)(int)viewer["enable"];
        m_viewer_width = (int)viewer["width"];
        m_viewer_height = (int)viewer["height"];
        
        if (m_enable_debug_output) {
            spdlog::info("[CONFIG] Viewer parameters:");
            spdlog::info("  - Enable: {}", m_viewer_enable);
            spdlog::info("  - Width: {}", m_viewer_width);
            spdlog::info("  - Height: {}", m_viewer_height);
        }
    }
    
    // Gravity Estimation Parameters
    cv::FileNode gravity_estimation = fs["gravity_estimation"];
    if (!gravity_estimation.empty()) {
        m_gravity_estimation_enable = (bool)(int)gravity_estimation["enable"];
        m_gravity_min_frames_for_estimation = (int)gravity_estimation["min_frames_for_estimation"];
        m_gravity_magnitude = (double)gravity_estimation["gravity_magnitude"];
        
        if (m_enable_debug_output) {
            spdlog::info("[CONFIG] Gravity estimation parameters:");
            spdlog::info("  - Enable: {}", m_gravity_estimation_enable);
            spdlog::info("  - Min frames: {}", m_gravity_min_frames_for_estimation);
            spdlog::info("  - Gravity magnitude: {:.3f} m/s²", m_gravity_magnitude);
        }
    }

    // IMU Noise Model Parameters
    cv::FileNode imu_noise = fs["imu_noise"];
    if (!imu_noise.empty()) {
        m_gyro_noise_density = (double)imu_noise["gyroscope_noise_density"];
        m_gyro_random_walk = (double)imu_noise["gyroscope_random_walk"];
        m_accel_noise_density = (double)imu_noise["accelerometer_noise_density"];
        m_accel_random_walk = (double)imu_noise["accelerometer_random_walk"];
        
        if (m_enable_debug_output) {
            spdlog::info("[CONFIG] IMU noise parameters loaded:");
            spdlog::info("  - Gyro noise density: {:.6e} rad/s/√Hz", m_gyro_noise_density);
            spdlog::info("  - Gyro random walk: {:.6e} rad/s²/√Hz", m_gyro_random_walk);
            spdlog::info("  - Accel noise density: {:.6e} m/s²/√Hz", m_accel_noise_density);
            spdlog::info("  - Accel random walk: {:.6e} m/s³/√Hz", m_accel_random_walk);
        }
    }
    
    // MapPoint Uncertainty Parameters
    cv::FileNode uncertainty = fs["uncertainty"];
    if (!uncertainty.empty()) {
        m_uncertainty_enable = (int)uncertainty["enable"] != 0;
        m_min_reprojection_error = (float)uncertainty["min_reprojection_error"];
        m_uncertainty_max_eigenvalue = (float)uncertainty["max_eigenvalue"];
        m_uncertainty_min_eigenvalue = (float)uncertainty["min_eigenvalue"];
        
        if (m_enable_debug_output) {
            spdlog::info("[CONFIG] Uncertainty parameters loaded:");
            spdlog::info("  - Uncertainty enable: {}", m_uncertainty_enable);
            spdlog::info("  - Min reprojection error: {:.2f} pixels", m_min_reprojection_error);
            spdlog::info("  - Max eigenvalue: {:.2f}", m_uncertainty_max_eigenvalue);
            spdlog::info("  - Min eigenvalue: {:.2f}", m_uncertainty_min_eigenvalue);
        }
    }
    
    // ⭐ RGBD Parameters (only loaded if camera type is RGBD)
    cv::FileNode rgbd = fs["rgbd"];
    if (!rgbd.empty()) {
        m_rgbd_depth_scale = (float)(double)rgbd["depth_scale"];
        
        // Uncertainty model
        if (!rgbd["uncertainty_a"].empty()) {
            m_rgbd_uncertainty_a = (float)(double)rgbd["uncertainty_a"];
        }
        if (!rgbd["uncertainty_b"].empty()) {
            m_rgbd_uncertainty_b = (float)(double)rgbd["uncertainty_b"];
        }
        if (!rgbd["uncertainty_c"].empty()) {
            m_rgbd_uncertainty_c = (float)(double)rgbd["uncertainty_c"];
        }
        
        // Feature selection
        if (!rgbd["enable_depth_quality_check"].empty()) {
            m_rgbd_enable_depth_quality_check = (bool)(int)rgbd["enable_depth_quality_check"];
        }
        if (!rgbd["min_depth_gradient_threshold"].empty()) {
            m_rgbd_min_depth_gradient_threshold = (float)(double)rgbd["min_depth_gradient_threshold"];
        }
        if (!rgbd["depth_consistency_window"].empty()) {
            m_rgbd_depth_consistency_window = (int)rgbd["depth_consistency_window"];
        }
        
        // Dense point cloud visualization
        if (!rgbd["enable_dense_cloud"].empty()) {
            m_rgbd_enable_dense_cloud = (bool)(int)rgbd["enable_dense_cloud"];
        }
        if (!rgbd["dense_cloud_stride"].empty()) {
            m_rgbd_dense_cloud_stride = (int)rgbd["dense_cloud_stride"];
        }
        if (!rgbd["vis_min_depth"].empty()) {
            m_rgbd_vis_min_depth = (float)(double)rgbd["vis_min_depth"];
        }
        if (!rgbd["vis_max_depth"].empty()) {
            m_rgbd_vis_max_depth = (float)(double)rgbd["vis_max_depth"];
        }
        
        if (m_enable_debug_output) {
            spdlog::info("[CONFIG] RGBD parameters loaded:");
            spdlog::info("  - Depth scale: {:.1f}", m_rgbd_depth_scale);
            spdlog::info("  - Visualization depth range: [{:.1f}, {:.1f}] m", m_rgbd_vis_min_depth, m_rgbd_vis_max_depth);
            spdlog::info("  - Dense cloud: {} (stride={}, RGB color mode)", 
                        m_rgbd_enable_dense_cloud, m_rgbd_dense_cloud_stride);
            spdlog::info("  - Uncertainty model: a={:.6f}, b={:.6f}, c={:.6f}", 
                        m_rgbd_uncertainty_a, m_rgbd_uncertainty_b, m_rgbd_uncertainty_c);
        }
    }
    
    // ========================================================================
    // Monocular Initialization Parameters
    // ========================================================================
    cv::FileNode initialization = fs["initialization"];
    if (!initialization.empty()) {
        m_init_min_parallax_pixels = (double)initialization["min_parallax_pixels"];
        
        m_init_ransac_threshold = (double)initialization["ransac_threshold"];
        m_init_ransac_confidence = (double)initialization["ransac_confidence"];
        m_init_ransac_max_iterations = (int)initialization["ransac_max_iterations"];
        
        m_init_min_triangulation_angle = (double)initialization["min_triangulation_angle"];
        m_init_max_reprojection_error = (double)initialization["max_reprojection_error"];
        
        if (m_enable_debug_output) {
            spdlog::info("[CONFIG] Monocular initialization parameters loaded:");
            spdlog::info("  - Min parallax: {:.1f} pixels (median)", m_init_min_parallax_pixels);
            spdlog::info("  - RANSAC: threshold={:.1f}px, confidence={:.2f}, max_iters={}", 
                        m_init_ransac_threshold, m_init_ransac_confidence, m_init_ransac_max_iterations);
            spdlog::info("  - Triangulation: min_angle={:.1f}°, max_reproj_error={:.1f}px",
                        m_init_min_triangulation_angle, m_init_max_reprojection_error);
        }
    }
    
    fs.release();
    return true;
}

} // namespace lightweight_vio
