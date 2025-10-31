/**
 * @file      Config.h
 * @brief     Defines a singleton class for managing configuration parameters.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-11
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

namespace lightweight_vio
{
    // Forward declaration
    class Camera;
    enum class CameraModel {
        PINHOLE,
        FISHEYE
    };

    enum class CameraType {
        STEREO,
        RGBD,
        MONOCULAR
    };

    class Config
    {
    public:
        static Config &getInstance()
        {
            static Config instance;
            return instance;
        }

        bool load(const std::string &config_file);

        // Functions for complex types only
        cv::TermCriteria term_criteria() const
        {
            return cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                    m_max_iterations, m_epsilon);
        }
        cv::TermCriteria stereo_term_criteria() const
        {
            return cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                    m_stereo_max_iterations, m_epsilon);
        }

        cv::Mat left_camera_matrix() const { return m_left_camera_matrix.clone(); }
        cv::Mat right_camera_matrix() const { return m_right_camera_matrix.clone(); }
        cv::Mat left_dist_coeffs() const { return m_left_dist_coeffs.clone(); }
        cv::Mat right_dist_coeffs() const { return m_right_dist_coeffs.clone(); }
        cv::Mat left_to_right_transform() const { return m_T_left_right.clone(); }
        cv::Mat left_T_BC() const { return m_T_left_BC.clone(); }
        cv::Mat right_T_BC() const { return m_T_right_BC.clone(); }
        
        // Camera model accessor
        CameraModel get_camera_model() const { return m_camera_model; }
        
        // ⭐ Camera type accessor
        CameraType get_camera_type() const { return m_camera_type; }
        bool is_stereo() const { return m_camera_type == CameraType::STEREO; }
        bool is_rgbd() const { return m_camera_type == CameraType::RGBD; }

        // Camera factory methods
        std::shared_ptr<Camera> create_left_camera() const;
        std::shared_ptr<Camera> create_right_camera() const;

        // Public member variables for simple access

        // Camera Model Parameters
        CameraModel m_camera_model = CameraModel::PINHOLE;
        CameraType m_camera_type = CameraType::STEREO;  // ⭐ Default: STEREO

        // Feature Detection Parameters
        int m_max_features = 150;
        double m_quality_level = 0.01;
        double m_min_distance = 30.0;

        // Grid-based Feature Distribution Parameters
        int m_grid_cols = 20;
        int m_grid_rows = 10;
        int m_max_features_per_grid = 4;

        // Optical Flow Parameters
        int m_window_size = 21;
        int m_max_level = 3;
        int m_max_iterations = 30;
        double m_epsilon = 0.01;
        double m_error_threshold = 30.0;
        double m_max_movement = 100.0;
        double m_min_eigen_threshold = 5e-2;
        double m_F_threshold = 1.0;  // Fundamental matrix RANSAC distance threshold (pixels)

        // Stereo Matching Parameters
        int m_stereo_window_size = 31;
        int m_stereo_max_level = 4;
        int m_stereo_max_iterations = 50;
        double m_stereo_min_eigen_threshold = 1e-3;

        double m_stereo_error_threshold = 50.0;
        double m_min_disparity = 0.1;
        double m_max_disparity = 300.0;
        double m_max_y_difference = 20.0;
        double m_epipolar_threshold = 5.0;

        // Stereo Rectification Parameters
        double m_max_rectified_y_difference = 2.0;

        // Global Depth Parameters
        double m_min_depth = 0.1;
        double m_max_depth = 100.0;


        // Keyframe Parameters
        double m_grid_coverage_ratio = 0.7;  // Add keyframe when coverage drops to this ratio of last keyframe's coverage
        int m_keyframe_window_size = 10;     // Number of keyframes to keep in sliding window
        double m_keyframe_time_threshold = 0.5;  // Force keyframe creation if time since last keyframe exceeds this (seconds)

        // PnP Optimization Parameters
        int m_pnp_max_iterations = 10;
        double m_pnp_function_tolerance = 1e-6;
        double m_pnp_gradient_tolerance = 1e-10;
        double m_pnp_parameter_tolerance = 1e-8;
        bool m_pnp_use_robust_kernel = true;
        // m_pnp_huber_delta_mono removed - hardcoded to 5.991
        bool m_pnp_enable_outlier_detection = true;
        int m_pnp_outlier_detection_rounds = 4;
       
        // Sliding Window Optimization Parameters  
        int m_sw_max_iterations = 10;
        double m_sw_function_tolerance = 1e-6;
        double m_sw_gradient_tolerance = 1e-10;
        double m_sw_parameter_tolerance = 1e-8;
        bool m_sw_use_robust_kernel = true;
        // m_sw_huber_delta removed - hardcoded to 5.991
        double m_sw_max_observation_weight = 3.0;

        // Legacy parameters (for backward compatibility)
        int m_pose_max_iterations = 10;
        double m_pose_function_tolerance = 1e-6;
        double m_pose_gradient_tolerance = 1e-10;
        double m_pose_parameter_tolerance = 1e-8;
        bool m_use_robust_kernel = true;
        // m_huber_delta_mono removed - hardcoded to 5.991
        // m_huber_delta_stereo removed - not used
        bool m_enable_outlier_detection = true;
        int m_outlier_detection_rounds = 3;
        double m_max_observation_weight = 3.0;

        // Camera Parameters
        int m_image_width = 752;
        int m_image_height = 480;
        int m_border_size = 1;

        // Performance Parameters
        bool m_enable_debug_output = false;

        // System Mode Parameters
        std::string m_system_mode = "VIO";  // "VO" or "VIO"
        
        // Viewer Parameters
        bool m_viewer_enable = false;       // Enable/disable 3D viewer
        int m_viewer_width = 1920;          // Viewer window width
        int m_viewer_height = 1080;         // Viewer window height
        
        // Gravity Estimation Parameters (VIO mode only)
        bool m_gravity_estimation_enable = true;
        int m_gravity_min_frames_for_estimation = 10;
        double m_gravity_magnitude = 9.81;

        // IMU Noise Model Parameters
        double m_gyro_noise_density = 1.6968e-04;      // rad/s/√Hz (gyro white noise)
        double m_gyro_random_walk = 1.9393e-05;        // rad/s²/√Hz (gyro bias diffusion)
        double m_accel_noise_density = 2.0000e-3;      // m/s²/√Hz (accel white noise)
        double m_accel_random_walk = 3.0000e-3;        // m/s³/√Hz (accel bias diffusion)

        // MapPoint Uncertainty Parameters
        bool m_uncertainty_enable = true;              // Enable uncertainty propagation for MapPoints
        float m_min_reprojection_error = 0.5f;         // Minimum reprojection error threshold for uncertainty calculation (pixels)
        float m_uncertainty_max_eigenvalue = 1.0f;     // Maximum eigenvalue limit for covariance scaling
        float m_uncertainty_min_eigenvalue = 0.0001f;  // Minimum eigenvalue limit for covariance scaling

        // ⭐ RGBD Specific Parameters
        float m_rgbd_depth_scale = 1000.0f;            // Depth conversion scale (default: mm to m)
        
        // RGBD uncertainty model: σ² = a*d² + b*d + c
        float m_rgbd_uncertainty_a = 0.0012f;          // Quadratic coefficient
        float m_rgbd_uncertainty_b = 0.0019f;          // Linear coefficient
        float m_rgbd_uncertainty_c = 0.0001f;          // Constant term
        
        // RGBD feature selection
        bool m_rgbd_enable_depth_quality_check = true; // Enable depth quality check
        float m_rgbd_min_depth_gradient_threshold = 0.1f; // Avoid depth discontinuities
        int m_rgbd_depth_consistency_window = 3;       // Depth consistency window size
        
        // RGBD dense point cloud visualization
        bool m_rgbd_enable_dense_cloud = true;         // Enable dense point cloud generation
        int m_rgbd_dense_cloud_stride = 4;             // Pixel sampling stride (1=all, 2=every 2nd, etc.)
        float m_rgbd_vis_min_depth = 0.3f;             // Minimum depth for heatmap visualization (clamp to red below this)
        float m_rgbd_vis_max_depth = 20.0f;            // Maximum depth for heatmap visualization (clamp to blue above this)

    private:
        Config() = default;

        // Private camera matrices (accessed through functions)
        cv::Mat m_left_camera_matrix;
        cv::Mat m_right_camera_matrix;
        cv::Mat m_left_dist_coeffs;
        cv::Mat m_right_dist_coeffs;
        cv::Mat m_T_left_right;
        cv::Mat m_T_left_BC;
        cv::Mat m_T_right_BC;
    };

} // namespace lightweight_vio