/**
 * @file      Optimizer.cpp
 * @brief     Implements pose and bundle adjustment optimizers using Ceres Solver.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "processing/Optimizer.h"
#include "processing/IMUHandler.h"  // 🎯 Complete type for IMUPreintegration
#include "database/Frame.h"
#include "database/MapPoint.h"
#include "optimization/Parameters.h"
#include "util/Config.h"
#include "optimization/Factors.h"
#include "database/Feature.h"
#include <spdlog/spdlog.h>
#include <sophus/se3.hpp>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <set>
#include <unordered_map>
#include <thread>

namespace lightweight_vio
{

    // Define global mutexes for thread-safe access
    std::mutex PnPOptimizer::s_mappoint_mutex;
    std::mutex PnPOptimizer::s_keyframe_mutex;

    PnPOptimizer::PnPOptimizer()
    {
    }

    OptimizationResult PnPOptimizer::optimize_pose(std::shared_ptr<Frame> frame)
    {
        OptimizationResult result;

        // Create Ceres problem
        ceres::Problem problem;

        // Convert frame pose to SE3 tangent space
        Eigen::Vector6d pose_params = frame_to_se3_tangent(frame);

        // Add parameter block first
        problem.AddParameterBlock(pose_params.data(), 6);

        // Set SE3 global parameterization for pose parameterization
        auto se3_global_param = new factor::SE3GlobalParameterization();
        problem.SetParameterization(pose_params.data(), se3_global_param);

        // Get camera parameters from frame
        double fx, fy, cx, cy;
        frame->get_camera_intrinsics(fx, fy, cx, cy);
        factor::CameraParameters camera_params(fx, fy, cx, cy);
        
        // // DEBUG: Print camera intrinsics
        // spdlog::debug("[DEBUG_CAM] Camera intrinsics: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}",
        //              fx, fy, cx, cy);

                // Add observations to the problem
        std::vector<ObservationInfo> observations;
        std::vector<int> feature_indices; // Track which features correspond to observations
        int num_valid_observations = 0;
        int num_excluded_outliers = 0;

        m_pnp_info_x_sqrt.clear();
        m_pnp_info_y_sqrt.clear();

        // Add mono PnP observations from frame's map points
        // Protect MapPoint access with mutex
        {
            std::lock_guard<std::mutex> lock(s_mappoint_mutex);
            const auto &map_points = frame->get_map_points();
            
            for (size_t i = 0; i < map_points.size(); ++i)
            {
                auto mp = map_points[i];
                if (!mp || mp->is_bad())
                {
                    continue;
                }

                // Give outliers a second chance - don't exclude them immediately
                // Only exclude if they've been consistently outliers for multiple frames
                // For now, let all features with map points participate in optimization
                bool is_previous_outlier = frame->get_outlier_flag(i);
                if (is_previous_outlier) {
                    // Still include but track that it was an outlier
                    num_excluded_outliers++; // This now means "previous outliers given another chance"
                }

                // Get 3D world point
                Eigen::Vector3d world_point = mp->get_position().cast<double>();

                // Get 2D observation from feature
                if (i >= frame->get_features().size())
                {
                    continue;
                }

                auto feature = frame->get_features()[i];

                // Get undistorted normalized coordinates and convert to pixel coordinates (consistent with CREATE_MP)
                cv::Point2f undistorted_pixel = feature->get_undistorted_coord();

                double undist_u = undistorted_pixel.x;
                double undist_v = undistorted_pixel.y;
                Eigen::Vector2d observation(undist_u, undist_v);

                // Add mono PnP observation with adaptive weighting based on config mode
                int num_observations = mp->get_observation_count();
                auto obs_info = add_observation(problem, pose_params.data(), world_point, observation, camera_params, mp, frame, 1.0);

                // Debug: Check if projection makes sense for first few features
                if (num_valid_observations < 3) {
                    // spdlog::debug("[PROJECTION] Feature {}: pixel=({:.2f},{:.2f}), world=({:.2f},{:.2f},{:.2f})", 
                    //              i, observation.x(), observation.y(), 
                    //              world_point.x(), world_point.y(), world_point.z());
                }

                if (obs_info.residual_id)
                {
                    observations.push_back(obs_info);
                    feature_indices.push_back(i);
                    num_valid_observations++;
                }
            }
        } // Release mutex here


        // spdlog::info("Min Max Mean of PnP info sqrt x : {}, {}, {}", 
        //              *std::min_element(m_pnp_info_x_sqrt.begin(), m_pnp_info_x_sqrt.end()),
        //              *std::max_element(m_pnp_info_x_sqrt.begin(), m_pnp_info_x_sqrt.end()),
        //              std::accumulate(m_pnp_info_x_sqrt.begin(), m_pnp_info_x_sqrt.end(), 0.0) / m_pnp_info_x_sqrt.size());

        // spdlog::info("Min Max Mean of PnP info sqrt y : {}, {}, {}", 
        //              *std::min_element(m_pnp_info_y_sqrt.begin(), m_pnp_info_y_sqrt.end()),
        //              *std::max_element(m_pnp_info_y_sqrt.begin(), m_pnp_info_y_sqrt.end()),
        //              std::accumulate(m_pnp_info_y_sqrt.begin(), m_pnp_info_y_sqrt.end(), 0.0) / m_pnp_info_y_sqrt.size());

        // Check if we have enough observations
        if (num_valid_observations < 5)
        {
            result.success = false;
            result.num_inliers = 0;
            return result;
        }

        // Get global config
        const auto& config = Config::getInstance();

        // Setup solver options
        ceres::Solver::Options options = setup_solver_options(config.m_pose_max_iterations);

        // Perform outlier detection rounds if enabled
        if (config.m_enable_outlier_detection)
        {
            double initial_cost = 0.0;
            double final_cost = 0.0;
            int total_iterations = 0;
            
            // Store initial pose parameters for resetting each round
            Eigen::Vector6d initial_pose_params = pose_params;
            
            for (int round = 0; round < config.m_outlier_detection_rounds; ++round)
            {
                // Reset pose to initial value for each round
                if (round > 0) {
                    pose_params = initial_pose_params;
                    // spdlog::debug("[POSE_OPT] Round {}: Reset pose to initial value", round);
                }
                
                // Solve
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                // Store costs for summary
                if (round == 0) {
                    initial_cost = summary.initial_cost;
                }
                final_cost = summary.final_cost;
                total_iterations += summary.iterations.size();

                // Detect outliers and update frame's outlier flags
                double *pose_data = pose_params.data();
                int num_inliers = detect_outliers(const_cast<double const *const *>(&pose_data), observations, feature_indices, frame);
                int num_outliers = observations.size() - num_inliers;

                // Remove outlier residual blocks for next iteration
                if (round < config.m_outlier_detection_rounds - 1)
                {
                    // Outliers are already disabled via set_outlier() in detect_outliers()
                    // The cost functions will return zero residuals and jacobians for outliers
                }

                // Update result
                result.initial_cost = initial_cost;
                result.final_cost = summary.final_cost;
                result.num_iterations += summary.iterations.size();
                result.success = (summary.termination_type == ceres::CONVERGENCE);
            }
            
            // Print consolidated optimization summary
            double *pose_data = pose_params.data();
            int final_inliers = detect_outliers(const_cast<double const *const *>(&pose_data), observations, feature_indices, frame);
            int final_outliers = observations.size() - final_inliers;
            
            // spdlog::info("[POSE_OPT] {} rounds: cost {:.3e} -> {:.3e}, {} iters, {} inliers/{} outliers", 
            //             config.m_outlier_detection_rounds, initial_cost, final_cost, 
            //             total_iterations, final_inliers, final_outliers);
                        
            // Detailed Ceres summary logging removed
        }
        else
        {
            // Single solve without outlier detection rounds
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            if (config.m_enable_outlier_detection) {
                // Perform outlier detection for final report
                double *pose_data = pose_params.data();
                int num_inliers = detect_outliers(const_cast<double const *const *>(&pose_data), observations, feature_indices, frame);
                int num_outliers = observations.size() - num_inliers;
                
                // spdlog::info("[POSE_OPT] Single solve: cost {:.3e} -> {:.3e}, {} iters, {} inliers/{} outliers", 
                //             summary.initial_cost, summary.final_cost, summary.iterations.size(),
                //             num_inliers, num_outliers);
            } else {
                // No outlier detection
                // spdlog::info("[POSE_OPT] Single solve: cost {:.3e} -> {:.3e}, {} iters, ALL {} features treated as inliers", 
                //             summary.initial_cost, summary.final_cost, summary.iterations.size(),
                //             observations.size());
            }

            result.success = (summary.termination_type == ceres::CONVERGENCE);
            result.initial_cost = summary.initial_cost;
            result.final_cost = summary.final_cost;
            result.num_iterations = summary.iterations.size();
            
            // Detailed Ceres summary logging removed
        }

        // Count final inliers/outliers and disconnect outlier map points based on config
        if (config.m_enable_outlier_detection) {
            result.num_inliers = 0;
            result.num_outliers = 0;
            int disconnected_map_points = 0;
            
            // Protect MapPoint disconnection with mutex
            std::lock_guard<std::mutex> lock(s_mappoint_mutex);
            const auto &outlier_flags = frame->get_outlier_flags();
            for (size_t i = 0; i < outlier_flags.size(); ++i)
            {
                bool is_outlier = outlier_flags[i];
                if (is_outlier)
                {
                    result.num_outliers++;
                    
                    // Disconnect outlier feature from its map point
                    auto map_point = frame->get_map_point(i);
                    if (map_point && !map_point->is_bad()) {
                        // Remove observation from map point
                        map_point->remove_observation(frame);
                        
                        // Remove map point from frame
                        frame->set_map_point(i, nullptr);
                        
                        disconnected_map_points++;
                    }
                }
                else
                {
                    result.num_inliers++;
                }
            }
            
            // if (disconnected_map_points > 0) {
            //     spdlog::warn("POSE_OPT[] Disconnected {} outlier map points", disconnected_map_points);
            // }
        } else {
            // Treat all observations as inliers when outlier detection is disabled
            result.num_inliers = observations.size();
            result.num_outliers = 0;
        }

        // Update result
        result.optimized_pose = se3_tangent_to_matrix(pose_params);


        // Update frame pose if optimization was successful
        if (result.success)
        {
            std::lock_guard<std::mutex> lock(s_keyframe_mutex);
            frame->set_Twb(result.optimized_pose);
        }

        // Summary is already printed in the optimization loop above
        // if (config.m_print_summary)
        // {
        //     spdlog::info("[POSE] Optimization: {} inliers, {} outliers", 
        //                 result.num_inliers, result.num_outliers);
        // }

        return result;
    }

   
    int PnPOptimizer::detect_outliers(double const *const *pose_params,
                                       const std::vector<ObservationInfo> &observations,
                                       const std::vector<int> &feature_indices,
                                       std::shared_ptr<Frame> frame)
    {
        int num_inliers = 0;

        // Chi-square threshold for 2DOF - use more relaxed threshold
        const double chi2_threshold = 5.991;  // Chi-square threshold for 2 DoF at 99% confidence

        // Collect chi2 values for statistics
        std::vector<double> inlier_chi2_values;
        std::vector<double> outlier_chi2_values;

        for (size_t i = 0; i < observations.size(); ++i)
        {
            // Use our custom Chi-square computation with information matrix
            double chi2_error = observations[i].cost_function->compute_chi_square(pose_params);

            // Mark as outlier if above threshold
            bool is_outlier = (chi2_error > chi2_threshold);
            int feature_idx = feature_indices[i];
            
            // Get previous outlier status (protect with keyframe mutex)
            bool was_outlier;
            {
                std::lock_guard<std::mutex> lock(s_keyframe_mutex);
                was_outlier = frame->get_outlier_flag(feature_idx);
                
                // Update outlier flag - can be both set and cleared based on current chi2 test
                frame->set_outlier_flag(feature_idx, is_outlier);
            }

            // Set outlier flag in the cost function to disable it for next optimization round
            observations[i].cost_function->set_outlier(is_outlier);

            if (!is_outlier)
            {
                num_inliers++;
                inlier_chi2_values.push_back(chi2_error);
                
                // Log recovery if this feature was previously an outlier
                if (was_outlier) {
                    // spdlog::debug("[POSE_OPT] Feature {} recovered from outlier (chi2: {:.3f})", 
                    //              feature_idx, chi2_error);
                }
            }
            else
            {
                outlier_chi2_values.push_back(chi2_error);
                
                // Log new outlier detection
                if (!was_outlier) {
                    // spdlog::debug("[POSE_OPT] Feature {} marked as outlier (chi2: {:.3f})", feature_idx, chi2_error);
                }
            }
        }

        // Print chi2 statistics (commented out to reduce log verbosity)
        // if (!inlier_chi2_values.empty())
        // {
        //     auto inlier_minmax = std::minmax_element(inlier_chi2_values.begin(), inlier_chi2_values.end());
        //     double inlier_sum = std::accumulate(inlier_chi2_values.begin(), inlier_chi2_values.end(), 0.0);
        //     double inlier_mean = inlier_sum / inlier_chi2_values.size();
        //     
        //     spdlog::info("[CHI2_STATS] Inliers ({}): min={:.3f}, max={:.3f}, mean={:.3f}", 
        //                 inlier_chi2_values.size(), *inlier_minmax.first, 
        //                 *inlier_minmax.second, inlier_mean);
        // }

        // if (!outlier_chi2_values.empty())
        // {
        //     auto outlier_minmax = std::minmax_element(outlier_chi2_values.begin(), outlier_chi2_values.end());
        //     double outlier_sum = std::accumulate(outlier_chi2_values.begin(), outlier_chi2_values.end(), 0.0);
        //     double outlier_mean = outlier_sum / outlier_chi2_values.size();
        //     
        //     spdlog::info("[CHI2_STATS] Outliers ({}): min={:.3f}, max={:.3f}, mean={:.3f}", 
        //                 outlier_chi2_values.size(), *outlier_minmax.first, 
        //                 *outlier_minmax.second, outlier_mean);
        // }

        // // Debug: Print details for some outliers to understand what's wrong
        // int debug_count = 0;
        // Eigen::Map<const Eigen::Vector6d> se3_tangent(pose_params[0]);
        // Sophus::SE3d current_pose = Sophus::SE3d::exp(se3_tangent);
        
        // // Convert matrix to string for logging
        // std::stringstream ss;
        // ss << current_pose.matrix();
        // // spdlog::debug("[OUTLIER_DEBUG] Current pose Twb:\n{}", ss.str());
        
        // for (size_t i = 0; i < observations.size() && debug_count < 3; ++i)
        // {
        //     double chi2_error = observations[i].cost_function->compute_chi_square(pose_params);
        //     if (chi2_error > chi2_threshold)
        //     {
        //         int feature_idx = feature_indices[i];
        //         auto feature = frame->get_features()[feature_idx];
        //         auto mp = frame->get_map_points()[feature_idx];
                
        //         // Manually project to see what the expected pixel should be
        //         Eigen::Vector3d world_pos = mp->get_position().cast<double>();
                
        //         // Transform to camera coordinates: Pc = Rcw * Pw + tcw
        //         Eigen::Matrix3d Rwb = current_pose.rotationMatrix();
        //         Eigen::Vector3d t_wb = current_pose.translation();
        //         Eigen::Matrix3d Rbw = Rwb.transpose();
        //         Eigen::Vector3d t_bw = -Rbw * t_wb;
                
        //         // Get T_cb (body-to-camera transform) from frame directly - CONSISTENT WITH add_observation
        //         const Eigen::Matrix4d& T_cb = frame->get_Tcb();
                
        //         // Transform to camera coordinates: Pc = T_cb * (Rbw * Pw + t_bw)
        //         Eigen::Vector3d point_body = Rbw * world_pos + t_bw;
        //         Eigen::Vector4d point_body_h(point_body.x(), point_body.y(), point_body.z(), 1.0);
        //         Eigen::Vector4d point_camera_h = T_cb * point_body_h;
        //         Eigen::Vector3d point_camera = point_camera_h.head<3>();
                
        //         // Project to image plane
        //         double fx, fy, cx, cy;
        //         frame->get_camera_intrinsics(fx, fy, cx, cy);
                
        //         if (point_camera.z() > 0) {
        //             double u_proj = fx * point_camera.x() / point_camera.z() + cx;
        //             double v_proj = fy * point_camera.y() / point_camera.z() + cy;
                    
        //             // Get undistorted observation using normalized coordinates (consistent with CREATE_MP)
        //             cv::Point2f undistorted_pixel = feature->get_undistorted_coord();
        //             double undist_u = undistorted_pixel.x;
        //             double undist_v = undistorted_pixel.y;

        //             auto map_point = frame->get_map_point(feature_idx);
        //             // spdlog::debug("[OUTLIER_DEBUG] Feature {}: chi2={:.3f}, observed_undist=({:.1f},{:.1f}), projected=({:.1f},{:.1f}), world=({:.2f},{:.2f},{:.2f})", 
        //             //              feature_idx, chi2_error, undist_u, undist_v,
        //             //              u_proj, v_proj, world_pos.x(), world_pos.y(), world_pos.z());
        //         } else {
        //             // spdlog::debug("[OUTLIER_DEBUG] Feature {}: chi2={:.3f}, BEHIND_CAMERA: z={:.2f}", 
        //             //              feature_idx, chi2_error, point_camera.z());
        //         }
        //         debug_count++;
        //     }
        // }

        return num_inliers;
    }

    ceres::Solver::Options PnPOptimizer::setup_solver_options(int max_iter) const
    {
        ceres::Solver::Options options;
        const auto& config = Config::getInstance();

        // Use provided max_iter directly
        options.max_num_iterations = max_iter;
        options.function_tolerance = config.m_pose_function_tolerance;
        options.gradient_tolerance = config.m_pose_gradient_tolerance;
        options.parameter_tolerance = config.m_pose_parameter_tolerance;

        // Use fixed solver configuration for now
        options.linear_solver_type = ceres::DENSE_QR;
        options.use_explicit_schur_complement = false;

        // Logging configuration - simplified (no config variables)
        options.logging_type = ceres::SILENT;
        options.minimizer_progress_to_stdout = false;

        return options;
    }

    Eigen::Vector6d PnPOptimizer::frame_to_se3_tangent(std::shared_ptr<Frame> frame) const
    {
        // Get frame pose (T_wb)
        Eigen::Matrix4f T_wb = frame->get_Twb();

        // Convert to double precision
        Eigen::Matrix4d T_wb_d = T_wb.cast<double>();

        // Fix numerical precision issues from float->double conversion
        Eigen::Matrix3d R = T_wb_d.block<3, 3>(0, 0);
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();
        
        // Ensure proper rotation (det(R) = 1)
        if (R.determinant() < 0) {
            R = -R;
        }
        
        // Reconstruct the pose matrix with orthogonalized rotation
        T_wb_d.block<3, 3>(0, 0) = R;

        // Now Sophus SE3 constructor will be happy
        Sophus::SE3d se3(T_wb_d);
        return se3.log();
    }

    Eigen::Matrix4f PnPOptimizer::se3_tangent_to_matrix(const Eigen::Vector6d &se3_tangent) const
    {
        // Convert tangent space to SE3 using Sophus (already guarantees proper SE3)
        Sophus::SE3d se3 = Sophus::SE3d::exp(se3_tangent);

        // Sophus already ensures proper SE3 structure, no need for SVD
        // Just convert to float at the end
        return se3.matrix().cast<float>();
    }

    ceres::LossFunction *PnPOptimizer::create_robust_loss(double delta) const
    {
        return new ceres::HuberLoss(delta);
    }

    Eigen::Matrix2d PnPOptimizer::create_information_matrix(double pixel_noise) const
    {
        // Information matrix is inverse of covariance matrix
        // For isotropic pixel noise: Covariance = sigma^2 * I
        // Information = (1/sigma^2) * I
        double precision = 1.0 / (pixel_noise * pixel_noise);
        return precision * Eigen::Matrix2d::Identity();
    }


    // Adaptive version with config-based mode selection
    ObservationInfo PnPOptimizer::add_observation(
        ceres::Problem &problem,
        double *pose_params,
        const Eigen::Vector3d &world_point,
        const Eigen::Vector2d &observation,
        const factor::CameraParameters &camera_params,
        std::shared_ptr<MapPoint> mappoint,
        std::shared_ptr<Frame> frame,
        const double pixel_noise_std)
    {
        // Get config instance to check information matrix mode
        const Config& config = Config::getInstance();
        
        // Create information matrix based on config mode
        Eigen::Matrix2d information;
        if (config.m_uncertainty_enable) {
            // Use adaptive information matrix based on MapPoint uncertainty
            information = create_information_from_uncertainty_propagation(mappoint, frame);
        } else 
        {
            // Use standard information matrix
            information = create_information_matrix(pixel_noise_std);
        }

        m_pnp_info_x_sqrt.push_back(sqrt(information(0, 0)));
        m_pnp_info_y_sqrt.push_back(sqrt(information(1, 1)));

        // Get T_cb (body-to-camera transform) from frame directly
        const Eigen::Matrix4d& T_cb = frame->get_Tcb();
        
        // Create mono PnP cost function with selected information matrix and T_cb
        auto cost_function = new factor::PnPFactor(observation, world_point, camera_params, T_cb, information);

        // Create robust loss function if enabled
        ceres::LossFunction *loss_function = nullptr;
        if (config.m_use_robust_kernel)
        {
            loss_function = create_robust_loss(sqrt(5.991));  // Chi-squared 95% threshold for 2 DOF
        }

        // Add residual block
        auto residual_id = problem.AddResidualBlock(
            cost_function, loss_function, pose_params);

        return ObservationInfo(residual_id, cost_function);
    }

// SlidingWindowOptimizer implementation

// Define global mutexes for SlidingWindowOptimizer
std::mutex SlidingWindowOptimizer::s_mappoint_mutex;
std::mutex SlidingWindowOptimizer::s_keyframe_mutex;

SlidingWindowOptimizer::SlidingWindowOptimizer(size_t window_size)
    : m_window_size(window_size), m_imu_enabled(false), m_gravity_magnitude(9.81)
{
    // Get config for initialization
    const Config& config = Config::getInstance();
    m_max_iterations = config.m_sw_max_iterations;  // Use sliding window specific max iterations
    m_huber_delta = sqrt(5.991);                          // Chi-squared 95% threshold for 2 DOF (hardcoded)
    m_pixel_noise_std = 1.0;                        // Default pixel noise
    m_outlier_threshold = 5.991;                    // Chi-square threshold for 2 DoF at 98% confidence (more relaxed)
    
    // Initialize gravity direction as default downward
    m_gravity_direction = Eigen::Vector3d(0.0, 0.0, -1.0);
}

SlidingWindowResult SlidingWindowOptimizer::optimize(
    const std::vector<std::shared_ptr<Frame>>& keyframes) {
    
    SlidingWindowResult result;
    
    
    if (keyframes.size() < 2) {
        return result;
    }
  
    
    auto map_points = collect_window_map_points(keyframes);
    
    if (map_points.empty()) {
        return result;
    }
    
    // Setup Ceres problem
    ceres::Problem problem;
    
    // Parameter storage for poses and map points
    std::vector<std::vector<double>> pose_params_vec(keyframes.size(), std::vector<double>(6));
    std::vector<std::vector<double>> point_params_vec(map_points.size(), std::vector<double>(3));
    
    // IMU parameter storage (only used if IMU is enabled)
    std::vector<std::vector<double>> velocity_params_vec;
    std::vector<double> accel_bias_params(3, 0.0);  // Shared bias across all keyframes
    std::vector<double> gyro_bias_params(3, 0.0);   // Shared bias across all keyframes
    std::vector<double> gravity_dir_params;
    
    // Setup visual optimization problem
    auto observations = setup_optimization_problem(
        problem, keyframes, map_points, pose_params_vec, point_params_vec);
    
    if (observations.empty()) {
        return result;
    }
    
    // Setup IMU parameter blocks and factors if enabled
    int num_imu_factors = 0;
    if (m_imu_enabled) {
        setup_imu_parameter_blocks(problem, keyframes, velocity_params_vec, 
                                  accel_bias_params, gyro_bias_params, gravity_dir_params);
        
        num_imu_factors = add_inertial_factors_to_sliding_window(
            problem, keyframes, pose_params_vec, velocity_params_vec,
            accel_bias_params, gyro_bias_params, gravity_dir_params);
    }
    
    // Configure solver options for two-stage optimization
    // First stage: Quick outlier detection with visual factors only (no IMU)
    int first_stage_max_iter = std::max(1, m_max_iterations / 2);
    ceres::Solver::Options first_stage_options = setup_solver_options(m_max_iterations);
    
    // Second stage: Precise optimization with all factors (visual + IMU if enabled)
    ceres::Solver::Options second_stage_options = setup_solver_options(m_max_iterations);
    
    ceres::Solver::Summary summary;
    
    // Initial cost evaluation
    double first_cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &first_cost, nullptr, nullptr, nullptr);

    // Stage 1: Quick optimization with more fixed keyframes for stability
    // Fix more keyframes in first stage for robust outlier detection
    int stage1_fixed_keyframes = 4;  // More conservative approach for first stage
    apply_marginalization_strategy(problem, keyframes, map_points, pose_params_vec, point_params_vec, stage1_fixed_keyframes);
    // spdlog::debug("[SlidingWindowOptimizer] Stage 1: Fixed {} keyframes for outlier detection", stage1_fixed_keyframes);
    
    ceres::Solve(first_stage_options, &problem, &summary);
    
    double stage1_cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &stage1_cost, nullptr, nullptr, nullptr);

    // Outlier detection phase: mark outliers but don't remove them yet
    for (const auto& obs_info : observations) {
        const double* pose_params = pose_params_vec[obs_info.keyframe_index].data();
        const double* point_params = point_params_vec[obs_info.mappoint_index].data();
        const double* params[2] = {pose_params, point_params};
        
        // Compute chi-square error
        double chi_square = obs_info.cost_function->compute_chi_square(params);
        
        // Mark as outlier if above threshold (equivalent to g2o's setLevel(1))
        bool is_outlier = (chi_square > m_outlier_threshold);
        obs_info.cost_function->set_outlier(is_outlier);
    }
    
    // Stage 2: Apply different marginalization strategy for precise optimization
    // Fix fewer keyframes in second stage for more degrees of freedom
    int stage2_fixed_keyframes = 1;  // More flexible approach for second stage
    apply_marginalization_strategy(problem, keyframes, map_points, pose_params_vec, point_params_vec, stage2_fixed_keyframes, true, true);
    // spdlog::debug("[SlidingWindowOptimizer] Stage 2: Reset constraints and fixed {} keyframes for precise optimization", stage2_fixed_keyframes);
    
    // Get cost before Stage 2 (after Stage 1 completion)
    double stage2_initial_cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &stage2_initial_cost, nullptr, nullptr, nullptr);
    
    // Stage 2: Precise optimization without robust kernel (full iterations)
    // Outliers are disabled via set_outlier - they return zero residuals
    ceres::Solver::Summary final_summary;
    ceres::Solve(second_stage_options, &problem, &final_summary);
    summary = final_summary; // Use final summary for cost reporting
    
    // Final cost evaluation (after Stage 2)
    double second_cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &second_cost, nullptr, nullptr, nullptr);
    result.num_iterations = summary.iterations.size();
    
    // Detect outliers and count inliers
    int num_inliers = detect_ba_outliers(
        pose_params_vec, point_params_vec, observations, keyframes, map_points);
    
    result.num_inliers = num_inliers;
    result.num_outliers = static_cast<int>(observations.size()) - num_inliers;
    result.num_poses_optimized = static_cast<int>(keyframes.size());
    result.num_points_optimized = static_cast<int>(map_points.size());
    
    // Check if Stage 2 optimization was successful using Brief Report costs
    bool stage2_cost_decreased = (summary.final_cost < summary.initial_cost);
    bool stage2_converged = (summary.termination_type == ceres::CONVERGENCE || summary.termination_type == ceres::USER_SUCCESS);
    
                    
    result.final_cost = summary.final_cost;
    result.initial_cost = summary.initial_cost;

    
    result.success = stage2_cost_decreased || stage2_converged;
    
    if (result.success) {
        // Update keyframes and map points with optimized values
        update_optimized_values(keyframes, map_points, pose_params_vec, point_params_vec);
        
        // Update observation point clouds after optimization
        auto uncertainty_start = std::chrono::high_resolution_clock::now();
        for (auto& mp : map_points) {
            if (mp && !mp->is_bad()) {
                // Update cached observation positions with optimized poses
                mp->update_uncertainty();
            }
        }
        auto uncertainty_end = std::chrono::high_resolution_clock::now();
        auto uncertainty_duration = std::chrono::duration_cast<std::chrono::microseconds>(uncertainty_end - uncertainty_start);
        double uncertainty_time_ms = uncertainty_duration.count() / 1000.0;
        // spdlog::info("[SlidingWindowOptimizer] Uncertainty update took {:.2f} ms for {} map points", 
        //              uncertainty_time_ms, map_points.size());
        
        // Update IMU states if IMU optimization is enabled
        if (m_imu_enabled && num_imu_factors > 0) {
            update_imu_optimized_values(keyframes, velocity_params_vec, 
                                       accel_bias_params, gyro_bias_params);
        }
        
        if (Config::getInstance().m_enable_debug_output) {
            spdlog::info("[SlidingWindowOptimizer] ✅ Optimization successful: {} poses, {} points, {} visual obs, {} IMU factors, {} inliers, {} outliers, cost: {:.10e} -> {:.10e}",
                        result.num_poses_optimized, result.num_points_optimized, 
                        observations.size(), num_imu_factors,
                        result.num_inliers, result.num_outliers, result.initial_cost, result.final_cost);
        }

    } else {
        spdlog::warn("[SlidingWindowOptimizer] ❌ Optimization failed (cost increased): {:.2e} -> {:.2e}, {}", result.initial_cost, result.final_cost, summary.BriefReport());
    }
    
    return result;
}

std::vector<std::shared_ptr<MapPoint>> SlidingWindowOptimizer::collect_window_map_points(
    const std::vector<std::shared_ptr<Frame>>& keyframes) const {
    
    std::set<std::shared_ptr<MapPoint>> unique_map_points;
    
    // Collect all unique map points from keyframes with mutex protection
    {
        std::lock_guard<std::mutex> lock(s_mappoint_mutex);
        for (const auto& keyframe : keyframes) {
            if (!keyframe) continue;
            
            const auto& map_points = keyframe->get_map_points();
            for (const auto& mp : map_points) {
                if (mp && !mp->is_bad()) {
                    unique_map_points.insert(mp);
                }
            }
        }
    }


    // Convert set to vector
    std::vector<std::shared_ptr<MapPoint>> result(unique_map_points.begin(), unique_map_points.end());
    
    // spdlog::info("[SlidingWindowOptimizer] Collected {} unique map points from {} keyframes",
    //             result.size(), keyframes.size());
    
    return result;
}

BAObservationInfo SlidingWindowOptimizer::add_observation(
    ceres::Problem& problem,
    double* pose_params,
    double* point_params,
    const Eigen::Vector2d& observation,
    const factor::CameraParameters& camera_params,
    std::shared_ptr<Frame> frame,
    std::shared_ptr<MapPoint> mappoint,
    int kf_index,
    int mp_index,
    double pixel_noise_std) {
    
    // Get T_CB transformation from frame
    Eigen::Matrix4d T_CB = frame->get_Tcb();
    
    // Get config instance to check information matrix mode
    const Config& config = Config::getInstance();
    
    // Create information matrix based on config mode
    Eigen::Matrix2d information;
    if (config.m_uncertainty_enable) {
        // Use adaptive information matrix based on MapPoint uncertainty
        information = create_information_from_uncertainty_propagation(mappoint, frame);
    } 
    else 
    {
        // Use standard information matrix
        information = create_information_matrix(pixel_noise_std);
    }

    m_sba_info_x_sqrt.push_back(sqrt(information(0, 0)));
    m_sba_info_y_sqrt.push_back(sqrt(information(1, 1)));

    // Create BA factor
    auto* cost_function = new factor::BAFactor(observation, camera_params, T_CB, information);
    
    // Create robust loss function
    ceres::LossFunction* loss_function = create_robust_loss(m_huber_delta);
    
    // Add residual block to problem
    ceres::ResidualBlockId residual_id = problem.AddResidualBlock(
        cost_function, loss_function, pose_params, point_params);
    
    return BAObservationInfo(residual_id, cost_function, kf_index, mp_index, information);
}


std::vector<BAObservationInfo> SlidingWindowOptimizer::setup_optimization_problem(
    ceres::Problem& problem,
    const std::vector<std::shared_ptr<Frame>>& keyframes,
    const std::vector<std::shared_ptr<MapPoint>>& map_points,
    std::vector<std::vector<double>>& pose_params_vec,
    std::vector<std::vector<double>>& point_params_vec) {
    
    std::vector<BAObservationInfo> observations;
    
    // Create map from MapPoint pointer to index for fast lookup
    std::unordered_map<std::shared_ptr<MapPoint>, int> mappoint_to_index;
    for (size_t i = 0; i < map_points.size(); ++i) {
        mappoint_to_index[map_points[i]] = static_cast<int>(i);
    }
    
    // Initialize pose parameters from keyframes with keyframe mutex protection
    {
        std::lock_guard<std::mutex> lock(s_keyframe_mutex);
        for (size_t kf_idx = 0; kf_idx < keyframes.size(); ++kf_idx) {
            const auto& keyframe = keyframes[kf_idx];
            Eigen::Matrix4f T_wb = keyframe->get_Twb();
            
            // Convert to double precision
            Eigen::Matrix4d T_wb_d = T_wb.cast<double>();
            
            // Extract rotation and translation
            Eigen::Matrix3d R_wb = T_wb_d.block<3, 3>(0, 0);
            Eigen::Vector3d t_wb = T_wb_d.block<3, 1>(0, 3);
            
            // Ensure rotation matrix is perfectly orthogonal using SVD
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_wb, Eigen::ComputeFullU | Eigen::ComputeFullV);
            R_wb = svd.matrixU() * svd.matrixV().transpose();
            
            // Ensure proper rotation (det = 1, not -1)
            if (R_wb.determinant() < 0) {
                Eigen::Matrix3d V_corrected = svd.matrixV();
                V_corrected.col(2) *= -1;  // Flip last column
                R_wb = svd.matrixU() * V_corrected.transpose();
            }
            
            // Reconstruct clean transformation matrix
            Eigen::Matrix4d T_wb_clean = Eigen::Matrix4d::Identity();
            T_wb_clean.block<3, 3>(0, 0) = R_wb;
            T_wb_clean.block<3, 1>(0, 3) = t_wb;
            
            // Convert to SE3 tangent space
            Sophus::SE3d se3_pose(T_wb_clean);
            Eigen::Vector6d tangent = se3_pose.log();
            
            std::copy(tangent.data(), tangent.data() + 6, pose_params_vec[kf_idx].data());
            
            // Add parameter block to problem FIRST
            problem.AddParameterBlock(pose_params_vec[kf_idx].data(), 6);
            
            // Then set pose parameterization
            auto* pose_parameterization = new factor::SE3GlobalParameterization();
            problem.SetParameterization(pose_params_vec[kf_idx].data(), pose_parameterization);
        }
    }
    
    // Initialize map point parameters
    for (size_t mp_idx = 0; mp_idx < map_points.size(); ++mp_idx) {
        const auto& map_point = map_points[mp_idx];
        Eigen::Vector3f position = map_point->get_position();
        
        point_params_vec[mp_idx][0] = position.x();
        point_params_vec[mp_idx][1] = position.y();
        point_params_vec[mp_idx][2] = position.z();
        
        // Add parameter block to problem FIRST
        problem.AddParameterBlock(point_params_vec[mp_idx].data(), 3);
        
        // Then set point parameterization
        auto* point_parameterization = new factor::MapPointParameterization();
        problem.SetParameterization(point_params_vec[mp_idx].data(), point_parameterization);
    }
    
    // Get camera parameters
    const Config& config = Config::getInstance();
    cv::Mat K = config.left_camera_matrix();
    factor::CameraParameters camera_params(
        K.at<double>(0, 0),  // fx
        K.at<double>(1, 1),  // fy
        K.at<double>(0, 2),  // cx
        K.at<double>(1, 2)   // cy
    );

    m_sba_info_x_sqrt.clear();
    m_sba_info_y_sqrt.clear();
    
    // Error statistics collection
    std::vector<double> reprojection_errors;
    std::vector<double> predicted_errors;
    
    // Add observations for each keyframe with mutex protection
    {
        std::lock_guard<std::mutex> lock(s_mappoint_mutex);
        for (size_t kf_idx = 0; kf_idx < keyframes.size(); ++kf_idx) {
            const auto& keyframe = keyframes[kf_idx];
            const auto& features = keyframe->get_features();
            const auto& frame_map_points = keyframe->get_map_points();
            
            for (size_t feat_idx = 0; feat_idx < features.size(); ++feat_idx) {
                const auto& feature = features[feat_idx];
                const auto& map_point = frame_map_points[feat_idx];
                
                // Skip invalid features or map points
                if (!feature || !feature->is_valid() || !map_point || map_point->is_bad()) {
                    continue;
                }
                
                // Skip outlier features
                if (keyframe->get_outlier_flag(feat_idx)) {
                    continue;
                }
                
                // Find map point index
                auto it = mappoint_to_index.find(map_point);
                if (it == mappoint_to_index.end()) {
                    continue; // Map point not in our optimization set
                }
                
                int mp_idx = it->second;
                
                // Get 2D observation using undistorted coordinates (consistent with PnP optimizer)
                cv::Point2f undistorted_pixel = feature->get_undistorted_coord();
                Eigen::Vector2d observation(undistorted_pixel.x, undistorted_pixel.y);
                
                // Add BA observation with observation-based information weighting
                int num_observations = map_point->get_observation_count();
                auto obs_info = add_observation(
                    problem,
                    pose_params_vec[kf_idx].data(),
                    point_params_vec[mp_idx].data(),
                    observation,
                    camera_params,
                    keyframe,
                    map_point,
                    static_cast<int>(kf_idx),
                    mp_idx,
                    m_pixel_noise_std);

                Eigen::Vector3f world_pos = map_point->get_position();
                Eigen::Matrix4f T_cw = keyframe->get_Twc().inverse();
                Eigen::Vector3f cam_pos = T_cw.block<3,3>(0,0) * world_pos + T_cw.block<3,1>(0,3);

                if (cam_pos.z() > 0) {
                    double fx = camera_params.fx;
                    double fy = camera_params.fy;
                    double cx = camera_params.cx;
                    double cy = camera_params.cy;

                    double u_proj = fx * cam_pos.x() / cam_pos.z() + cx;
                    double v_proj = fy * cam_pos.y() / cam_pos.z() + cy;

                    double reproj_error = std::sqrt(std::pow(u_proj - observation.x(), 2) + std::pow(v_proj - observation.y(), 2));

                    Eigen::Matrix2d cov = obs_info.information_matrix.inverse();
                    double sigma_u = std::sqrt(cov(0,0));
                    double sigma_v = std::sqrt(cov(1,1));
                    double predicted_error = std::sqrt(sigma_u*sigma_u + sigma_v*sigma_v);

                    // Collect statistics
                    reprojection_errors.push_back(reproj_error);
                    predicted_errors.push_back(predicted_error);
                } 

                observations.push_back(obs_info);
            }
        }
    }
    
    // // Print error statistics
    // if (!reprojection_errors.empty() && !predicted_errors.empty()) {
    //     // Calculate statistics for reprojection errors
    //     std::sort(reprojection_errors.begin(), reprojection_errors.end());
    //     double reproj_min = reprojection_errors.front();
    //     double reproj_max = reprojection_errors.back();
    //     double reproj_mean = std::accumulate(reprojection_errors.begin(), reprojection_errors.end(), 0.0) / reprojection_errors.size();
    //     double reproj_median = reprojection_errors[reprojection_errors.size() / 2];
        
    //     // Calculate statistics for predicted errors
    //     std::sort(predicted_errors.begin(), predicted_errors.end());
    //     double pred_min = predicted_errors.front();
    //     double pred_max = predicted_errors.back();
    //     double pred_mean = std::accumulate(predicted_errors.begin(), predicted_errors.end(), 0.0) / predicted_errors.size();
    //     double pred_median = predicted_errors[predicted_errors.size() / 2];
        
    //     spdlog::info("[SlidingWindow] Error Statistics ({} observations):", reprojection_errors.size());
    //     spdlog::info("  Reprojection - Min: {:.3f}, Max: {:.3f}, Mean: {:.3f}, Median: {:.3f}", 
    //                  reproj_min, reproj_max, reproj_mean, reproj_median);
    //     spdlog::info("  Predicted    - Min: {:.3f}, Max: {:.3f}, Mean: {:.3f}, Median: {:.3f}", 
    //                  pred_min, pred_max, pred_mean, pred_median);
    //     spdlog::info("  Mean Ratio (Reproj/Pred): {:.3f}", reproj_mean / pred_mean);
    // }


    // spdlog::info("Min Max Mean of sqrt information x : {}, {}, {}", 
    //               *std::min_element(m_sba_info_x_sqrt.begin(), m_sba_info_x_sqrt.end()),
    //               *std::max_element(m_sba_info_x_sqrt.begin(), m_sba_info_x_sqrt.end()),
    //               std::accumulate(m_sba_info_x_sqrt.begin(), m_sba_info_x_sqrt.end(), 0.0) / m_sba_info_x_sqrt.size());

    // spdlog::info("Min Max Mean of sqrt information y : {}, {}, {}", 
    //               *std::min_element(m_sba_info_y_sqrt.begin(), m_sba_info_y_sqrt.end()),
    //               *std::max_element(m_sba_info_y_sqrt.begin(), m_sba_info_y_sqrt.end()),
    //               std::accumulate(m_sba_info_y_sqrt.begin(), m_sba_info_y_sqrt.end(), 0.0) / m_sba_info_y_sqrt.size());

    // spdlog::info("[SlidingWindowOptimizer] Setup problem: {} keyframes, {} map points, {} observations",
    //             keyframes.size(), map_points.size(), observations.size());
    
    return observations;
}

void SlidingWindowOptimizer::apply_marginalization_strategy(
    ceres::Problem &problem,
    const std::vector<std::shared_ptr<Frame>> &keyframes,
    const std::vector<std::shared_ptr<MapPoint>> &map_points,
    const std::vector<std::vector<double>> &pose_params_vec,
    const std::vector<std::vector<double>> &point_params_vec,
    int num_fixed_keyframes,
    bool reset_constraints,
    bool marginalize_points)
{

    if (keyframes.empty()) return;
    
    // Reset existing constraints if requested
    if (reset_constraints) {
        // Make all pose parameters variable first
        for (size_t i = 0; i < pose_params_vec.size(); ++i) {
            problem.SetParameterBlockVariable(const_cast<double*>(pose_params_vec[i].data()));
        }
        // Make all point parameters variable first
        for (size_t i = 0; i < point_params_vec.size(); ++i) {
            problem.SetParameterBlockVariable(const_cast<double*>(point_params_vec[i].data()));
        }
    }
    
    // Ensure num_fixed_keyframes is within valid range
    int max_fixed = std::min(num_fixed_keyframes, static_cast<int>(keyframes.size()));
    max_fixed = std::max(max_fixed, 1); // At least fix one keyframe for gauge freedom
    
    // Fix the first N keyframes as reference to prevent gauge freedom
    for (int i = 0; i < max_fixed && i < static_cast<int>(pose_params_vec.size()); ++i) {
        problem.SetParameterBlockConstant(const_cast<double*>(pose_params_vec[i].data()));
        // spdlog::debug("[SlidingWindowOptimizer] Fixed keyframe {} (index {}) as reference",
        //              keyframes[i]->get_frame_id(), i);
    }
    
    
    // Optional: Fix map points with insufficient observations or high precision
    int fixed_points = 0;
    int fixed_by_low_obs = 0;
    int fixed_by_low_eigenvalue = 0;
    int marginalized_points = 0;
    for (size_t mp_idx = 0; mp_idx < map_points.size(); ++mp_idx) {
        const auto& map_point = map_points[mp_idx];
        int obs_count = map_point->get_observation_count();
        
        bool should_fix = false;
        bool fixed_by_eigenvalue = false;

        // Fix map points with too few observations
        if (obs_count < 3)
        {
            should_fix = true;
        }


        if (should_fix) {
            problem.SetParameterBlockConstant(const_cast<double*>(point_params_vec[mp_idx].data()));
            fixed_points++;
            
        } 
    }

}

int SlidingWindowOptimizer::detect_ba_outliers(
    const std::vector<std::vector<double>>& pose_params_vec,
    const std::vector<std::vector<double>>& point_params_vec,
    const std::vector<BAObservationInfo>& observations,
    const std::vector<std::shared_ptr<Frame>>& keyframes,
    const std::vector<std::shared_ptr<MapPoint>>& map_points) {
    
    int num_inliers = 0;
    std::set<int> outlier_map_point_indices; // Track which map points are outliers
    std::vector<double> chi2_values;
    
    for (const auto& obs_info : observations) {
        // Get parameter pointers
        const double* pose_params = pose_params_vec[obs_info.keyframe_index].data();
        const double* point_params = point_params_vec[obs_info.mappoint_index].data();
        
        const double* params[2] = {pose_params, point_params};
        
        // Compute chi-square error
        double chi_square = obs_info.cost_function->compute_chi_square(params);
        chi2_values.push_back(chi_square);
        
        // Check against threshold
        bool is_inlier = (chi_square <= m_outlier_threshold);
        if (is_inlier) {
            num_inliers++;
        } else {
            // Mark this map point index as outlier
            outlier_map_point_indices.insert(obs_info.mappoint_index);
        }
        
        // Mark outlier in cost function (will return zero residuals)
        obs_info.cost_function->set_outlier(!is_inlier);
    }
    
    // Mark all outlier map points as bad and disconnect from all frames
    int marked_bad = 0;
    int disconnected_features = 0;
    
    // Protect MapPoint modifications and keyframe outlier flags with mutexes
    {
        std::lock_guard<std::mutex> mp_lock(s_mappoint_mutex);
        std::lock_guard<std::mutex> kf_lock(s_keyframe_mutex);
        
        for (int mp_idx : outlier_map_point_indices) {
            if (mp_idx >= 0 && mp_idx < static_cast<int>(map_points.size())) {
                auto map_point = map_points[mp_idx];
                if (map_point && !map_point->is_bad()) {
                    // First, find all frames that observe this map point and mark their features as outliers
                    for (const auto& keyframe : keyframes) {
                        const auto& frame_map_points = keyframe->get_map_points();
                        for (size_t feat_idx = 0; feat_idx < frame_map_points.size(); ++feat_idx) {
                            if (frame_map_points[feat_idx] == map_point) {
                                // Mark this feature as outlier in the frame
                                keyframe->set_outlier_flag(feat_idx, true);
                                // Remove the map point connection
                                keyframe->set_map_point(feat_idx, nullptr);
                                disconnected_features++;
                            }
                        }
                    }
                    
                    // Then mark the map point as bad
                    map_point->set_bad();
                    marked_bad++;
                }
            }
        }
    }
    
    // Log chi-square statistics
    if (!chi2_values.empty()) {
        auto minmax = std::minmax_element(chi2_values.begin(), chi2_values.end());
        double mean = std::accumulate(chi2_values.begin(), chi2_values.end(), 0.0) / chi2_values.size();
        
    }
    
   
    return num_inliers;
}

void SlidingWindowOptimizer::update_optimized_values(
    const std::vector<std::shared_ptr<Frame>>& keyframes,
    const std::vector<std::shared_ptr<MapPoint>>& map_points,
    const std::vector<std::vector<double>>& pose_params_vec,
    const std::vector<std::vector<double>>& point_params_vec) {
    
    int updated_keyframes = 0;
    int updated_map_points = 0;
    
    // Update keyframe poses with keyframe mutex protection
    {
        std::lock_guard<std::mutex> lock(s_keyframe_mutex);
        for (size_t kf_idx = 0; kf_idx < keyframes.size(); ++kf_idx) {
            const auto& keyframe = keyframes[kf_idx];
            if (!keyframe) continue;
            
            const auto& pose_params = pose_params_vec[kf_idx];
            
            // Store original pose for comparison
            Eigen::Matrix4f original_pose = keyframe->get_Twb();
            
            // Convert SE3 tangent space back to matrix
            Eigen::Map<const Eigen::Vector6d> tangent(pose_params.data());
            Sophus::SE3d se3_pose = Sophus::SE3d::exp(tangent);
            Eigen::Matrix4f T_wb = se3_pose.matrix().cast<float>();
            
            // Check if pose actually changed
            Eigen::Matrix4f pose_diff = T_wb - original_pose;
            double pose_change = pose_diff.norm();
            
            keyframe->set_Twb(T_wb);
            updated_keyframes++;
            
           
        }
    }
    
    // Update map point positions with mutex protection
    {
        std::lock_guard<std::mutex> lock(s_mappoint_mutex);
        
        auto uncertainty_start = std::chrono::high_resolution_clock::now();
        int uncertainty_updates = 0;
        
        for (size_t mp_idx = 0; mp_idx < map_points.size(); ++mp_idx) {
            const auto& map_point = map_points[mp_idx];
            if (!map_point || map_point->is_bad()) continue;
            
            const auto& point_params = point_params_vec[mp_idx];
            
            // Store original position for comparison
            Eigen::Vector3f original_pos = map_point->get_position();
            
            Eigen::Vector3f new_position(
                static_cast<float>(point_params[0]),
                static_cast<float>(point_params[1]),
                static_cast<float>(point_params[2]));
            
            // Check if position actually changed
            Eigen::Vector3f pos_diff = new_position - original_pos;
            double position_change = pos_diff.norm();

            map_point->set_position(new_position);
            updated_map_points++;

            auto observations = map_point->get_observations();

            for (auto &obs : observations)
            {
                auto frame = obs.frame.lock();
                if (!frame)
                    continue;

                int feature_idx = obs.feature_index;

                // Get the feature and update its depth
                auto &features = frame->get_features();

                auto feature = features[feature_idx];

                // Get world position and transform to camera coordinates
                Eigen::Vector3f world_pos = map_point->get_position();
                auto P_cam = frame->get_Twc().inverse() * world_pos.homogeneous();

                // // Update depth in the observation
                // float learning_rate = 0.1f;
                // float new_depth = learning_rate * P_cam.z() + (1.0f - learning_rate) * feature->get_depth();

                float new_depth = P_cam.z();

                // Check if depth is within valid range using config
                const auto &config = Config::getInstance();
                if (new_depth <= config.m_min_depth || new_depth >= config.m_max_depth)
                {

                    feature->set_depth(-1.0f);
                    continue; // Skip invalid depth values
                }


                // spdlog::info("Depth update for MapPoint {} in Frame {}: {:.8f} -> {:.8f}", 
                //             map_point->get_id(), frame->get_frame_id(), feature->get_depth(), new_depth);

                feature->set_depth(new_depth);


                frame->set_depth(feature_idx, new_depth);
                uncertainty_updates++;
            }

            // map_point->update_uncertainty();
        }

        // if (Config::getInstance().m_uncertainty_enable && uncertainty_updates > 0) {
        //     auto uncertainty_end = std::chrono::high_resolution_clock::now();
        //     auto uncertainty_duration = std::chrono::duration_cast<std::chrono::microseconds>(uncertainty_end - uncertainty_start);
        //     double avg_time_per_update = static_cast<double>(uncertainty_duration.count()) / uncertainty_updates;

        //     spdlog::info("[UNCERTAINTY_TIMING] Updated {} MapPoints uncertainty in {:.3f}ms (avg: {:.3f}μs per point)",
        //                 uncertainty_updates, uncertainty_duration.count() / 1000.0, avg_time_per_update);
        // }
    }
    
    // spdlog::info("[UPDATE] Updated {} keyframes and {} map points", 
    //             updated_keyframes, updated_map_points);
}

ceres::Solver::Options SlidingWindowOptimizer::setup_solver_options(int max_iter) const {
    ceres::Solver::Options options;
    const Config& config = Config::getInstance();
    
    // Use sparse solver for bundle adjustment
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    
    // Use provided max_iter directly
    options.max_num_iterations = max_iter;
    options.function_tolerance = config.m_sw_function_tolerance;
    options.gradient_tolerance = config.m_sw_gradient_tolerance;
    options.parameter_tolerance = config.m_sw_parameter_tolerance;
    
    // Enable detailed logging if needed
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;
    
    // Use multiple threads if available
    options.num_threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
    
    return options;
}

ceres::LossFunction* SlidingWindowOptimizer::create_robust_loss(double delta) const {
    return new ceres::HuberLoss(delta);
}

Eigen::Matrix2d SlidingWindowOptimizer::create_information_matrix(double pixel_noise) const {
    Eigen::Matrix2d information_matrix;
    double variance = pixel_noise * pixel_noise;
    information_matrix << 1.0 / variance, 0.0,
                         0.0, 1.0 / variance;
    return information_matrix;
}

Eigen::Matrix2d SlidingWindowOptimizer::create_information_from_uncertainty_propagation(
    std::shared_ptr<MapPoint> mappoint,
    std::shared_ptr<Frame> frame) const
{

    // Get world uncertainty from MapPoint (3x3 covariance matrix)
    Eigen::Matrix3d world_uncertainty = mappoint->get_world_uncertainty().cast<double>();
    Eigen::Matrix2d information_matrix = mappoint->transform_uncertainty_world_to_pixel(world_uncertainty.cast<float>(), frame).cast<double>().inverse();

    return information_matrix + Eigen::Matrix2d::Identity(); // Add small value to diagonal for numerical stability
}

// ===============================================================================
// INERTIAL OPTIMIZER IMPLEMENTATION
// ===============================================================================

InertialOptimizer::InertialOptimizer() {
    // Load optimization parameters from config if available
    const auto& config = Config::getInstance();
    
    // Use existing optimization parameters from config
    m_params.max_iterations = config.m_pnp_max_iterations;
    m_params.function_tolerance = config.m_pnp_function_tolerance;
    m_params.gradient_tolerance = config.m_pnp_gradient_tolerance;
    m_params.parameter_tolerance = config.m_pnp_parameter_tolerance;
    m_params.use_robust_kernel = config.m_pnp_use_robust_kernel;
}

InertialOptimizationResult InertialOptimizer::optimize_imu_initialization(
    std::vector<Frame*>& frames,
    std::shared_ptr<IMUHandler> imu_handler) {
    
    InertialOptimizationResult result;
    
    if (frames.size() < 5) {
        spdlog::warn("[IMU_INIT] Need at least 5 frames for IMU initialization");
        return result;
    }
    
    if (!imu_handler || !imu_handler->is_initialized()) {
        spdlog::warn("[IMU_INIT] IMU handler not initialized");
        return result;
    }
    
    
    
    
    // ===============================================================================
    // STEP 1: Setup optimization parameters - separate velocity and bias arrays
    // ===============================================================================
    
    std::vector<std::vector<double>> pose_params_vec(frames.size(), std::vector<double>(6));
    std::vector<std::vector<double>> velocity_params_vec(frames.size(), std::vector<double>(3));  // velocity (3D)
    std::vector<std::vector<double>> accel_bias_params_vec(frames.size(), std::vector<double>(3)); // accel bias (3D)  
    std::vector<std::vector<double>> gyro_bias_params_vec(frames.size(), std::vector<double>(3));  // gyro bias (3D)
    std::vector<double> gravity_dir_params(2, 0.0); // 2D gravity direction
    
    setup_imu_init_vertices(frames, imu_handler, pose_params_vec, velocity_params_vec, 
                           accel_bias_params_vec, gyro_bias_params_vec, gravity_dir_params);
    
    // ===============================================================================
    // LOG INITIAL STATES (BEFORE OPTIMIZATION) - Minimized
    // ===============================================================================
    

    
    // ===============================================================================
    // STEP 2: Setup Ceres problem
    // ===============================================================================
    
    ceres::Problem problem;
    ceres::Solver::Options options;
    
    // IMU initialization requires very strict convergence for accurate bias estimation
    options.max_num_iterations = 100;  // Much longer optimization for better convergence
    // Solver configuration optimized for IMU initialization
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = false;  // Disable progress output
    options.logging_type = ceres::SILENT;
    // options.check_gradients = true;  // Disable gradient checking temporarily to allow optimization


    
    // ===============================================================================
    // STEP 3: Add parameter blocks and parameterizations
    // ===============================================================================
    
    // Add pose parameter blocks (fixed during IMU initialization)
    for (size_t i = 0; i < pose_params_vec.size(); ++i) {
        problem.AddParameterBlock(pose_params_vec[i].data(), 6);
        auto* pose_parameterization = new factor::SE3GlobalParameterization();
        problem.SetParameterization(pose_params_vec[i].data(), pose_parameterization);
        problem.SetParameterBlockConstant(pose_params_vec[i].data()); // Fix poses
    }
    
    // Add velocity parameter blocks (3D each) - temporarily fixed
    for (size_t i = 0; i < velocity_params_vec.size(); ++i) {
        problem.AddParameterBlock(velocity_params_vec[i].data(), 3);
        // problem.SetParameterBlockConstant(velocity_params_vec[i].data()); // Fix velocity
    }
    
    // Add accelerometer bias parameter blocks (3D each) - temporarily fixed
    for (size_t i = 0; i < accel_bias_params_vec.size(); ++i) {
        problem.AddParameterBlock(accel_bias_params_vec[i].data(), 3);
        // problem.SetParameterBlockConstant(accel_bias_params_vec[i].data()); // Fix accel bias
    }
    
    // Add gyroscope bias parameter blocks (3D each) - temporarily fixed
    for (size_t i = 0; i < gyro_bias_params_vec.size(); ++i) {
        problem.AddParameterBlock(gyro_bias_params_vec[i].data(), 3);
        // problem.SetParameterBlockConstant(gyro_bias_params_vec[i].data()); // Fix gyro bias
    }
    
    // Add gravity direction parameter block
    problem.AddParameterBlock(gravity_dir_params.data(), 2);
    // problem.SetParameterBlockConstant(gravity_dir_params.data()); // Fix gravity direction during IMU init
    
    // ===============================================================================
    // STEP 4: Add InertialGravityFactor factors
    // ===============================================================================
    
    int inertial_gravity_factors_added = add_inertial_gravity_factors(
        problem, frames, imu_handler,
        pose_params_vec, velocity_params_vec, accel_bias_params_vec, gyro_bias_params_vec, gravity_dir_params);
    
    if (inertial_gravity_factors_added == 0) {
        spdlog::error("[IMU_INIT] No inertial gravity factors added");
        return result;
    }
    
    
    // ===============================================================================
    // STEP 5: Add bias priors for regularization
    // ===============================================================================
    
    add_imu_init_priors(problem, frames, velocity_params_vec, accel_bias_params_vec, gyro_bias_params_vec);
    
    // ===============================================================================
    // STEP 6: Solve optimization
    // ===============================================================================
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // ===============================================================================
    // STEP 7: Extract optimized results
    // ===============================================================================
    
    result.success = (summary.termination_type == ceres::CONVERGENCE || summary.termination_type == ceres::USER_SUCCESS);
    result.num_iterations = summary.iterations.size();
    result.initial_cost = summary.initial_cost;
    result.final_cost = summary.final_cost;
    result.cost_reduction = result.initial_cost - result.final_cost;
    
    if (result.success) {
        // Recover optimized states
        recover_imu_init_states(frames, imu_handler, pose_params_vec, velocity_params_vec, 
                               accel_bias_params_vec, gyro_bias_params_vec, gravity_dir_params, result.Tgw_init);

        if (Config::getInstance().m_enable_debug_output)
        {
            spdlog::info("🎯 [IMU_INIT] Optimization Complete:");
            spdlog::info("  ✅ Success: {}", result.success);
            spdlog::info("  ⏱️  Duration: {} ms", duration.count());
            spdlog::info("  🔄 Iterations: {}", result.num_iterations);
            spdlog::info("  📊 Cost: {:.6f} → {:.6f} (Δ={:.6f})",
                         result.initial_cost, result.final_cost, result.cost_reduction);
            spdlog::info("  📋 InertialGravity factors: {}", inertial_gravity_factors_added);

            // Print optimized gravity direction
            spdlog::info("  🌍 Optimized Gravity Dir: ({:.6f}, {:.6f})",
                         gravity_dir_params[0], gravity_dir_params[1]);
        }
    } else {
        spdlog::error("❌ [IMU_INIT] Optimization failed: {}", summary.BriefReport());
    }
    
    return result;
}

void InertialOptimizer::setup_imu_init_vertices(
    const std::vector<Frame*>& frames,
    std::shared_ptr<IMUHandler> imu_handler,
    std::vector<std::vector<double>>& pose_params_vec,
    std::vector<std::vector<double>>& velocity_params_vec,
    std::vector<std::vector<double>>& accel_bias_params_vec,
    std::vector<std::vector<double>>& gyro_bias_params_vec,
    std::vector<double>& gravity_dir_params) {
    

    // spdlog::info("🔧 [IMU_INIT] Setting up IMU initialization vertices...");
    // Skip first frame (index 0) - only use frames 1,2,3,4... for IMU initialization
    size_t num_frames_for_optimization = frames.size() - 1;
    
    if (num_frames_for_optimization == 0) {
        // spdlog::error("[IMU_INIT] No frames available for optimization after skipping first frame");
        return;
    }
    
    // spdlog::info("🔄 [IMU_INIT] Using frames 1-{} for optimization (skipping first keyframe)", frames.size() - 1);
    
    // Resize parameter vectors for optimization frames only (excluding first frame)
    pose_params_vec.resize(num_frames_for_optimization, std::vector<double>(6));
    velocity_params_vec.resize(num_frames_for_optimization, std::vector<double>(3));  // velocity (3D)
    accel_bias_params_vec.resize(num_frames_for_optimization, std::vector<double>(3)); // accel bias (3D)
    gyro_bias_params_vec.resize(num_frames_for_optimization, std::vector<double>(3));  // gyro bias (3D)
    
    // Setup pose parameters for optimization frames (frames[1] to frames[n-1])
    for (size_t opt_idx = 0; opt_idx < num_frames_for_optimization; ++opt_idx) {
        size_t frame_idx = opt_idx + 1; // Skip first frame: frames[1], frames[2], ...
        auto* frame = frames[frame_idx];
        
        // spdlog::info("🎯 [IMU_INIT] Processing Frame[{}] (ID: {}) -> OptIdx[{}]", frame_idx, frame->get_frame_id(), opt_idx);
        
        // Initialize pose parameters using SE3 tangent space
        Eigen::Matrix4f Twb = frame->get_Twb();
        Eigen::Matrix4d Twb_d = Twb.cast<double>();
        
        // Extract rotation and translation
        Eigen::Matrix3d R_wb = Twb_d.block<3, 3>(0, 0);
        Eigen::Vector3d t_wb = Twb_d.block<3, 1>(0, 3);
        
        // Ensure rotation matrix is perfectly orthogonal using SVD
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_wb, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R_wb = svd.matrixU() * svd.matrixV().transpose();
        
        // Ensure proper rotation (det = 1, not -1)
        if (R_wb.determinant() < 0) {
            Eigen::Matrix3d V_corrected = svd.matrixV();
            V_corrected.col(2) *= -1;  // Flip last column
            R_wb = svd.matrixU() * V_corrected.transpose();
        }
        
        // Reconstruct clean transformation matrix
        Eigen::Matrix4d T_wb_clean = Eigen::Matrix4d::Identity();
        T_wb_clean.block<3, 3>(0, 0) = R_wb;
        T_wb_clean.block<3, 1>(0, 3) = t_wb;
        
        // Convert to SE3 tangent space
        Sophus::SE3d se3_pose(T_wb_clean);
        Eigen::Vector6d tangent = se3_pose.log();
        
        std::copy(tangent.data(), tangent.data() + 6, pose_params_vec[opt_idx].data());
        
        // Initialize velocity+bias parameters [v(3), ba(3), bg(3)]
        // Try to initialize velocity from all available preintegration sources
        Eigen::Vector3f frame_velocity = Eigen::Vector3f::Zero();
        
        // First, try to initialize velocity using the improved frame method
        frame->initialize_velocity_from_preintegration();
        frame_velocity = frame->get_velocity();
        
        // If frame initialization didn't work, try direct calculation
        if (frame_velocity.norm() < 1e-6) {
            double dt = frame->get_dt_from_last_keyframe();
            auto preintegration = frame->get_imu_preintegration_from_last_keyframe();
            
            if (preintegration && dt > 0.001 && dt < 1.0) {
                // Direct calculation as fallback (delta_V is already velocity, don't divide by time!)
                frame_velocity = frame->get_Twb().block<3,3>(0,0) * preintegration->delta_V;
                frame->set_velocity(frame_velocity);
                
                spdlog::info("🔄 [IMU_INIT] OptFrame[{}] (FrameID: {}): Used fallback direct calculation velocity=({:.4f}, {:.4f}, {:.4f})", 
                             opt_idx, frame->get_frame_id(), frame_velocity.x(), frame_velocity.y(), frame_velocity.z());
            } else {
                // Still no valid data, keep zero
                frame_velocity = Eigen::Vector3f::Zero();
                spdlog::warn("⚠️  [IMU_INIT] OptFrame[{}] (FrameID: {}): No valid preintegration data, using zero velocity", 
                             opt_idx, frame->get_frame_id());
            }
        }
        
        // Store the computed velocity in the frame for future use
        frame->set_velocity(frame_velocity);
        
        // Separate velocity and bias parameters
        velocity_params_vec[opt_idx][0] = static_cast<double>(frame_velocity.x());  // vx
        velocity_params_vec[opt_idx][1] = static_cast<double>(frame_velocity.y());  // vy  
        velocity_params_vec[opt_idx][2] = static_cast<double>(frame_velocity.z());  // vz
        
        accel_bias_params_vec[opt_idx][0] = 0.0;  // ba_x (accel bias)
        accel_bias_params_vec[opt_idx][1] = 0.0;  // ba_y  
        accel_bias_params_vec[opt_idx][2] = 0.0;  // ba_z
        
        gyro_bias_params_vec[opt_idx][0] = 0.0;  // bg_x (gyro bias)
        gyro_bias_params_vec[opt_idx][1] = 0.0;  // bg_y
        gyro_bias_params_vec[opt_idx][2] = 0.0;  // bg_z
        
    }
    
    // Initialize gravity direction to "down" (small perturbations around z-down)
    gravity_dir_params[0] = 0.0; // Small x rotation
    gravity_dir_params[1] = 0.0; // Small y rotation
    
    // spdlog::info("🏁 [IMU_INIT] Initialized {} pose vertices, {} velocity+bias vertices", 
    //              pose_params_vec.size(), velocity_bias_params_vec.size());
    // spdlog::info("📊 [IMU_INIT] Strategy: Frame[0]=ZERO_VEL, Frame[1-{}]=MULTI_SOURCE_AVERAGED_VEL + PER_FRAME_BIAS", 
    //              frames.size() - 1);
}

int InertialOptimizer::add_inertial_gravity_factors(
    ceres::Problem& problem,
    const std::vector<Frame*>& frames,
    std::shared_ptr<IMUHandler> imu_handler,
    const std::vector<std::vector<double>>& pose_params_vec,
    const std::vector<std::vector<double>>& velocity_params_vec,
    const std::vector<std::vector<double>>& accel_bias_params_vec,
    const std::vector<std::vector<double>>& gyro_bias_params_vec,
    const std::vector<double>& gravity_dir_params) {
    
    int factors_added = 0;
    
    // Add InertialGravityFactor factors between consecutive optimization frames
    // Note: pose_params_vec and velocity_bias_params_vec only contain optimization frames (excluding first keyframe)
    size_t num_opt_frames = pose_params_vec.size();
    
    for (size_t opt_idx = 0; opt_idx < num_opt_frames - 1; ++opt_idx) {
        // Map optimization indices to actual frame indices (skip first frame)
        size_t frame_i_idx = opt_idx + 1;      // frames[1], frames[2], ...
        size_t frame_j_idx = frame_i_idx + 1;  // frames[2], frames[3], ...
        
        Frame* frame_i = frames[frame_i_idx];
        Frame* frame_j = frames[frame_j_idx];
        
        // Use the pre-calculated dt from keyframe creation for frame_j
        double dt = frame_j->get_dt_from_last_keyframe();
        
        if (dt < 0.001 || dt > 1.0) {
            spdlog::warn("[IMU_INIT] Invalid dt={:.6f}s between frames {} and {}", 
                         dt, frame_i->get_frame_id(), frame_j->get_frame_id());
            continue;
        }
        
        // Use ACTUAL stored preintegration from frame_j (from frame_i to frame_j)
        auto preintegration = frame_j->get_imu_preintegration_from_last_keyframe();
        
        if (!preintegration) {
            spdlog::warn("[IMU_INIT] No stored preintegration available for frame {} -> {}, skipping factor", 
                         frame_i->get_frame_id(), frame_j->get_frame_id());
            continue;
        }
        
        
        // Create InertialGravityFactor
        double gravity_magnitude = 9.81; // Standard gravity
        auto* inertial_gravity_factor = new factor::InertialGravityFactor(preintegration, gravity_magnitude);
        
        // Create Huber loss for IMU factor
        // IMU measurements can have outliers, especially during rapid motion
        // Chi-square(15 DOF, 99%) = 16.63 for 15 degrees of freedom at 99% confidence
        double imu_huber_delta = sqrt(16.63);  // 15 DOF, 99% 
        auto* imu_loss_function = new ceres::HuberLoss(imu_huber_delta);
        
        // Add residual block using separate parameter arrays (7 parameter version) with Huber loss
        // NOTE: InertialGravityFactor expects [pose1, velocity1, GYRO_bias, ACCEL_bias, pose2, velocity2, gravity_dir]
        problem.AddResidualBlock(inertial_gravity_factor, imu_loss_function,
                                const_cast<double*>(pose_params_vec[opt_idx].data()),           // pose1 (6D)
                                const_cast<double*>(velocity_params_vec[opt_idx].data()),       // velocity1 (3D)
                                const_cast<double*>(gyro_bias_params_vec[opt_idx].data()),      // GYRO_bias1 (3D) - parameters[2]
                                const_cast<double*>(accel_bias_params_vec[opt_idx].data()),     // ACCEL_bias1 (3D) - parameters[3]
                                const_cast<double*>(pose_params_vec[opt_idx+1].data()),         // pose2 (6D)
                                const_cast<double*>(velocity_params_vec[opt_idx+1].data()),     // velocity2 (3D)
                                const_cast<double*>(gravity_dir_params.data()));                // gravity_dir (2D)
        
        factors_added++;
    }
    
    // spdlog::info("📊 [IMU_INIT] Total InertialGravityFactor factors added: {}", factors_added);
    
    return factors_added;
}

void InertialOptimizer::add_imu_init_priors(
    ceres::Problem& problem,
    const std::vector<Frame*>& frames,
    const std::vector<std::vector<double>>& velocity_params_vec,
    const std::vector<std::vector<double>>& accel_bias_params_vec,
    const std::vector<std::vector<double>>& gyro_bias_params_vec) {
    
    // Add velocity+bias priors for each optimization frame (excluding first keyframe)
    for (size_t opt_idx = 0; opt_idx < velocity_params_vec.size(); ++opt_idx) {
        size_t frame_idx = opt_idx + 1; // Convert optimization index to actual frame index
        auto* frame = frames[frame_idx];
        
        // Create velocity+bias prior [v(3), ba(3), bg(3)]
        Eigen::VectorXd velocity_bias_prior(9);
        
        // Use ACTUAL preintegration velocity as prior (not zero!) - includes gravity effects
        Eigen::Vector3f frame_velocity = frame->get_velocity();

        velocity_bias_prior[0] = static_cast<double>(frame_velocity.x());
        velocity_bias_prior[1] = static_cast<double>(frame_velocity.y());
        velocity_bias_prior[2] = static_cast<double>(frame_velocity.z());
        
        // Zero priors for biases
        velocity_bias_prior[3] = 0.0; // ba_x
        velocity_bias_prior[4] = 0.0; // ba_y
        velocity_bias_prior[5] = 0.0; // ba_z
        velocity_bias_prior[6] = 0.0; // bg_x
        velocity_bias_prior[7] = 0.0; // bg_y
        velocity_bias_prior[8] = 0.0; // bg_z
        
        // Information matrix (9x9) - different weights for velocity and biases
        Eigen::MatrixXd information = Eigen::MatrixXd::Zero(9, 9);
        double velocity_weight = 0.01;  // Small velocity prior weight
        double bias_weight = 1.0;      // Stronger bias prior weight 
        
        // Set diagonal elements
        information(0, 0) = velocity_weight; // vx
        information(1, 1) = velocity_weight; // vy
        information(2, 2) = velocity_weight; // vz
        information(3, 3) = bias_weight;     // ba_x
        information(4, 4) = bias_weight;     // ba_y
        information(5, 5) = bias_weight;     // ba_z
        information(6, 6) = bias_weight;     // bg_x
        information(7, 7) = bias_weight;     // bg_y
        information(8, 8) = bias_weight;     // bg_z
        
        // Create separate priors for velocity and biases
        Eigen::Vector3d velocity_prior(velocity_bias_prior[0], velocity_bias_prior[1], velocity_bias_prior[2]);
        Eigen::Vector3d accel_bias_prior(velocity_bias_prior[3], velocity_bias_prior[4], velocity_bias_prior[5]); 
        Eigen::Vector3d gyro_bias_prior(velocity_bias_prior[6], velocity_bias_prior[7], velocity_bias_prior[8]);
        
        // Information matrices (3x3 each)
        Eigen::Matrix3d velocity_info = Eigen::Matrix3d::Identity() * velocity_weight;
        Eigen::Matrix3d accel_bias_info = Eigen::Matrix3d::Identity() * bias_weight;
        Eigen::Matrix3d gyro_bias_info = Eigen::Matrix3d::Identity() * bias_weight;
        
        // Create cost functions
        auto* velocity_prior_cost = new factor::VectorPriorFactor<3>(velocity_prior, velocity_info);
        auto* accel_bias_prior_cost = new factor::VectorPriorFactor<3>(accel_bias_prior, accel_bias_info);
        auto* gyro_bias_prior_cost = new factor::VectorPriorFactor<3>(gyro_bias_prior, gyro_bias_info);
        
        // Add residual blocks for separate parameters
        problem.AddResidualBlock(velocity_prior_cost, nullptr, const_cast<double*>(velocity_params_vec[opt_idx].data()));
        problem.AddResidualBlock(accel_bias_prior_cost, nullptr, const_cast<double*>(accel_bias_params_vec[opt_idx].data()));
        problem.AddResidualBlock(gyro_bias_prior_cost, nullptr, const_cast<double*>(gyro_bias_params_vec[opt_idx].data()));
        
    }
    
    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("📌 [IMU_INIT] Added velocity+bias priors for {} frames", velocity_params_vec.size());
    }
}

void InertialOptimizer::recover_imu_init_states(
    const std::vector<Frame*>& frames,
    std::shared_ptr<IMUHandler> imu_handler,
    const std::vector<std::vector<double>>& pose_params_vec,
    const std::vector<std::vector<double>>& velocity_params_vec,
    const std::vector<std::vector<double>>& accel_bias_params_vec,
    const std::vector<std::vector<double>>& gyro_bias_params_vec,
    const std::vector<double>& gravity_dir_params,
    Eigen::Matrix4f& T_gw) {
    
    // ===============================================================================
    // STEP 7.1: Update frame velocities and biases with optimized values (silently)
    // ===============================================================================
    
    // Store initial states for comparison
    std::vector<Eigen::Vector3f> initial_velocities;
    
    // Get initial states before updating
    for (size_t opt_idx = 0; opt_idx < velocity_params_vec.size(); ++opt_idx) {
        size_t frame_idx = opt_idx + 1;
        auto* frame = frames[frame_idx];
        initial_velocities.push_back(frame->get_velocity());
    }
    
    // Update frame poses and velocities+biases with optimized values for optimization frames only
    for (size_t opt_idx = 0; opt_idx < velocity_params_vec.size(); ++opt_idx) {
        size_t frame_idx = opt_idx + 1; // Convert optimization index to actual frame index
        auto* frame = frames[frame_idx];
        
        // Extract velocity and biases from separate arrays
        Eigen::Vector3f optimized_velocity(
            static_cast<float>(velocity_params_vec[opt_idx][0]),
            static_cast<float>(velocity_params_vec[opt_idx][1]),
            static_cast<float>(velocity_params_vec[opt_idx][2]));
        
        Eigen::Vector3f optimized_accel_bias(
            static_cast<float>(accel_bias_params_vec[opt_idx][0]),
            static_cast<float>(accel_bias_params_vec[opt_idx][1]),
            static_cast<float>(accel_bias_params_vec[opt_idx][2]));
            
        Eigen::Vector3f optimized_gyro_bias(
            static_cast<float>(gyro_bias_params_vec[opt_idx][0]),
            static_cast<float>(gyro_bias_params_vec[opt_idx][1]),
            static_cast<float>(gyro_bias_params_vec[opt_idx][2]));
        
        frame->set_velocity(optimized_velocity);
        frame->set_accel_bias(optimized_accel_bias);
        frame->set_gyro_bias(optimized_gyro_bias);
    }
    
    // Compute and log optimized gravity vector (silently)
    double theta_x = gravity_dir_params[0];
    double theta_y = gravity_dir_params[1];
    
    Eigen::Matrix3d R_x = Eigen::AngleAxisd(theta_x, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Matrix3d R_y = Eigen::AngleAxisd(theta_y, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d R_wg = R_y * R_x;
    
    Eigen::Vector3d g_I(0, 0, -9.81);  // gravity in gravity frame
    Eigen::Vector3d g_world = R_wg * g_I;  // gravity in world frame
    
    // ===============================================================================
    // POST-OPTIMIZATION PROCESSING: Apply results and transform to gravity frame
    // ===============================================================================
    
    // 1. Initialize first keyframe velocity and bias from optimized frames (silently)
    if (frames.size() >= 2) {
        Eigen::Vector3f frame1_velocity = frames[1]->get_velocity();
        frames[0]->set_velocity(frame1_velocity);
    }
    
    // 2. Compute average bias from ONLY optimized frames (Frame[1], Frame[2], Frame[3]) (silently)
    Eigen::Vector3f avg_gyro_bias = Eigen::Vector3f::Zero();
    Eigen::Vector3f avg_accel_bias = Eigen::Vector3f::Zero();
    int bias_count = 0;
    
    // Only use frames that were actually optimized and have constraints (Frame[1], Frame[2], Frame[3])
    for (size_t opt_idx = 0; opt_idx < velocity_params_vec.size() - 1; ++opt_idx) { // Exclude last frame (Frame[4])
        size_t frame_idx = opt_idx + 1; // Frame[1], Frame[2], Frame[3]
        auto* frame = frames[frame_idx];
        avg_gyro_bias += frame->get_gyro_bias();
        avg_accel_bias += frame->get_accel_bias();
        bias_count++;
    }
    
    if (bias_count > 0) {
        avg_gyro_bias /= static_cast<float>(bias_count);
        avg_accel_bias /= static_cast<float>(bias_count);
    }
    
    // 3. Apply averaged bias to ALL 5 keyframes (Frame[0] through Frame[4]) (silently)
    for (size_t i = 0; i < frames.size(); ++i) {
        frames[i]->set_accel_bias(avg_accel_bias);
        frames[i]->set_gyro_bias(avg_gyro_bias);
    }
    
    // 4. Update IMUHandler's global bias with the computed average (silently)
    imu_handler->set_bias(avg_gyro_bias, avg_accel_bias);
    
    // 5. Update all preintegrations with the unified averaged bias for all frames (silently)
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> frame_biases;
    for (size_t i = 1; i < frames.size(); ++i) { // All frames from Frame[1] to Frame[4] get same averaged bias
        frame_biases.emplace_back(avg_gyro_bias, avg_accel_bias); // Use unified averaged bias
    }
    imu_handler->update_preintegrations_with_optimized_bias({frames.begin() + 1, frames.end()}, frame_biases);
    
    // 6. Set optimized gravity in IMUHandler (silently)
    imu_handler->set_gravity(g_world.cast<float>());
    
    // Store velocities before gravity transformation
    std::vector<Eigen::Vector3f> velocities_before_transform;
    for (size_t i = 0; i < frames.size(); ++i) {
        velocities_before_transform.push_back(frames[i]->get_velocity());
    }
    
    // 7. Transform all keyframes and map points to gravity-aligned frame (silently)
    // Collect all map points from keyframes for transformation
    std::vector<std::shared_ptr<MapPoint>> all_map_points;
    std::set<std::shared_ptr<MapPoint>> unique_map_points;
    
    for (const auto& frame : frames) {
        const auto& frame_map_points = frame->get_map_points();
        for (const auto& mp : frame_map_points) {
            if (mp && !mp->is_bad()) {
                unique_map_points.insert(mp);
            }
        }
    }
    
    all_map_points.assign(unique_map_points.begin(), unique_map_points.end());
    
    // Gravity transformation - reduced logging
    if (imu_handler->is_gravity_aligned()) {
        // Convert raw pointers to shared_ptr for the transform function
        std::vector<std::shared_ptr<Frame>> shared_frames;
        for (Frame* frame : frames) {
            if (frame) {
                shared_frames.push_back(std::shared_ptr<Frame>(frame, [](Frame*){})); // Non-owning shared_ptr
            }
        }

        bool transform_success = imu_handler->transform_to_gravity_frame(shared_frames, all_map_points, T_gw);
        if (transform_success)
        {
            imu_handler->set_gravity_aligned_coordinate_system();
            if (Config::getInstance().m_enable_debug_output)
            {
                spdlog::info("[IMU_INIT] ✅ Gravity alignment applied to {} frames and {} map points",
                             frames.size(), all_map_points.size());
            }
        }
    }

    // ===============================================================================
    // 📋 FINAL RESULTS: Show only key information
    // ===============================================================================

    // Frame processing starts
    if (Config::getInstance().m_enable_debug_output) {
    
    spdlog::info("� [IMU_OPTIMIZATION_RESULTS]");
    spdlog::info("  ✅ Optimization completed successfully");
    
    // Final IMU bias (unified across all frames)
    spdlog::info("  🔧 Final IMU Bias: Gyro=({:.10f}, {:.10f}, {:.6f}), Accel=({:.6f}, {:.6f}, {:.6f})",
                 avg_gyro_bias.x(), avg_gyro_bias.y(), avg_gyro_bias.z(),
                 avg_accel_bias.x(), avg_accel_bias.y(), avg_accel_bias.z());
    }
    
}

void InertialOptimizer::setup_params(
    const std::vector<Frame*>& frames,
    std::vector<std::vector<double>>& pose_params_vec,
    std::vector<std::vector<double>>& velocity_params_vec,
    std::vector<double>& gyro_bias_params,
    std::vector<double>& accel_bias_params) {
    
    pose_params_vec.clear();
    velocity_params_vec.clear();
    pose_params_vec.resize(frames.size(), std::vector<double>(6));
    velocity_params_vec.resize(frames.size(), std::vector<double>(3));
    
    // Setup pose and velocity parameters for each frame
    for (size_t i = 0; i < frames.size(); ++i) {
        auto* frame = frames[i];
        
        // Initialize pose parameters
        Eigen::Matrix4f Twb = frame->get_Twb();
        Eigen::Vector3d translation = Twb.block<3,1>(0,3).cast<double>();
        Eigen::Matrix3d rotation = Twb.block<3,3>(0,0).cast<double>();
        
        // Ensure rotation matrix is perfectly orthogonal using SVD
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
        rotation = svd.matrixU() * svd.matrixV().transpose();
        
        // Ensure proper rotation (det = 1, not -1)
        if (rotation.determinant() < 0) {
            Eigen::Matrix3d V_corrected = svd.matrixV();
            V_corrected.col(2) *= -1;  // Flip last column
            rotation = svd.matrixU() * V_corrected.transpose();
        }
        
        // SE(3) parameterization
        Sophus::SE3d se3(rotation, translation);
        auto tangent = se3.log();  // SE3d::Tangent type
        
        pose_params_vec[i][0] = tangent[0]; pose_params_vec[i][1] = tangent[1]; pose_params_vec[i][2] = tangent[2];
        pose_params_vec[i][3] = tangent[3]; pose_params_vec[i][4] = tangent[4]; pose_params_vec[i][5] = tangent[5];
        
        // Get current frame velocity for initialization
        Eigen::Vector3f current_velocity = frame->get_velocity();
        
        // Initialize velocity parameters with current frame velocity
        velocity_params_vec[i][0] = static_cast<double>(current_velocity.x());
        velocity_params_vec[i][1] = static_cast<double>(current_velocity.y()); 
        velocity_params_vec[i][2] = static_cast<double>(current_velocity.z());
        
        // Log initial parameter setup
        spdlog::debug("[SETUP_PARAMS] Frame[{}]: pos=({:.4f},{:.4f},{:.4f}), vel=({:.4f},{:.4f},{:.4f})",
                     i, translation.x(), translation.y(), translation.z(),
                     current_velocity.x(), current_velocity.y(), current_velocity.z());
    }
    
    // Initialize biases to zero
    gyro_bias_params[0] = 0.0; gyro_bias_params[1] = 0.0; gyro_bias_params[2] = 0.0;
    accel_bias_params[0] = 0.0; accel_bias_params[1] = 0.0; accel_bias_params[2] = 0.0;

    spdlog::info("🎯 [SETUP_PARAMS] Initialized {} frames with current velocities and zero biases", frames.size());
}



void InertialOptimizer::recover_optimized_states(
    const std::vector<Frame*>& frames,
    const std::vector<std::vector<double>>& pose_params_vec,
    const std::vector<std::vector<double>>& velocity_params_vec,
    const double* gyro_bias_params,
    const double* accel_bias_params) {
    
    spdlog::info("🔄 [RECOVER_STATES] Applying optimized parameters to frames:");
    
    // Recover optimized velocities and update frames
    for (size_t i = 0; i < frames.size(); ++i) {
        // Store original velocity for comparison
        Eigen::Vector3f original_velocity = frames[i]->get_velocity();
        
        // Extract optimized velocity from parameters
        Eigen::Vector3f optimized_velocity(
            static_cast<float>(velocity_params_vec[i][0]),
            static_cast<float>(velocity_params_vec[i][1]),
            static_cast<float>(velocity_params_vec[i][2]));
        
        // Calculate velocity change
        Eigen::Vector3f velocity_change = optimized_velocity - original_velocity;
        
        // Update frame velocity
        frames[i]->set_velocity(optimized_velocity);
        
        // Log detailed comparison
        spdlog::info("  Frame[{}]: vel BEFORE=({:.4f}, {:.4f}, {:.4f}) → AFTER=({:.4f}, {:.4f}, {:.4f}) [Δ={:.4f}]",
                     i,
                     original_velocity.x(), original_velocity.y(), original_velocity.z(),
                     optimized_velocity.x(), optimized_velocity.y(), optimized_velocity.z(),
                     velocity_change.norm());
    }
    
    // Update IMU handler biases if significant changes
    Eigen::Vector3f optimized_gyro_bias(
        static_cast<float>(gyro_bias_params[0]),
        static_cast<float>(gyro_bias_params[1]),
        static_cast<float>(gyro_bias_params[2]));
    
    Eigen::Vector3f optimized_accel_bias(
        static_cast<float>(accel_bias_params[0]),
        static_cast<float>(accel_bias_params[1]),
        static_cast<float>(accel_bias_params[2]));
    
    // Log bias parameter details
    spdlog::info("  📊 Optimized Bias Parameters:");
    spdlog::info("    Gyro Bias: ({:.6f}, {:.6f}, {:.6f}) [norm: {:.6f}]",
                 optimized_gyro_bias.x(), optimized_gyro_bias.y(), optimized_gyro_bias.z(),
                 optimized_gyro_bias.norm());
    spdlog::info("    Accel Bias: ({:.6f}, {:.6f}, {:.6f}) [norm: {:.6f}]",
                 optimized_accel_bias.x(), optimized_accel_bias.y(), optimized_accel_bias.z(),
                 optimized_accel_bias.norm());
    
    spdlog::info("✅ [RECOVER_STATES] Successfully updated {} frame velocities and bias parameters", frames.size());
}

void InertialOptimizer::cleanup_vertices(
    std::vector<std::vector<double>>& pose_params_vec,
    std::vector<std::vector<double>>& velocity_params_vec) {
    
    // Clear parameter vectors (automatic cleanup for std::vector)
    pose_params_vec.clear();
    velocity_params_vec.clear();
}

// ============================================================================
// SlidingWindowOptimizer IMU Methods
// ============================================================================

void SlidingWindowOptimizer::enable_imu_optimization(
    std::shared_ptr<IMUHandler> imu_handler,
    const Eigen::Vector3d& gravity_direction) {
    
    m_imu_enabled = true;
    m_imu_handler = imu_handler;
    m_gravity_direction = gravity_direction.normalized();
    
    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("[SW_IMU] IMU optimization enabled with gravity direction: ({:.3f}, {:.3f}, {:.3f})",
                     m_gravity_direction.x(), m_gravity_direction.y(), m_gravity_direction.z());
    }
}

void SlidingWindowOptimizer::disable_imu_optimization() {
    m_imu_enabled = false;
    m_imu_handler.reset();
    
    spdlog::info("[SW_IMU] IMU optimization disabled, using visual-only mode");
}

int SlidingWindowOptimizer::add_inertial_factors_to_sliding_window(
    ceres::Problem& problem,
    const std::vector<std::shared_ptr<Frame>>& keyframes,
    const std::vector<std::vector<double>>& pose_params_vec,
    const std::vector<std::vector<double>>& velocity_params_vec,
    const std::vector<double>& accel_bias_params,
    const std::vector<double>& gyro_bias_params,
    const std::vector<double>& gravity_dir_params) {
    
    if (!m_imu_enabled || !m_imu_handler) {
        return 0;
    }
    
    int factors_added = 0;
    
    // Add InertialGravityFactor between consecutive keyframes using existing implementation
    for (size_t i = 0; i < keyframes.size() - 1; ++i) {
        auto frame_i = keyframes[i];
        auto frame_j = keyframes[i + 1];
        
        // Get preintegration data from frame_j (from frame_i to frame_j)
        auto preintegration = frame_j->get_imu_preintegration_from_last_keyframe();
        
        if (!preintegration) {
            continue;
        }
        
        // Create InertialGravityFactor (reusing existing implementation)
        auto* inertial_gravity_factor = new factor::InertialGravityFactor(preintegration, m_gravity_magnitude);
  
        double imu_huber_delta = sqrt(16.63);  // 15 DOF, 99%
        auto* imu_loss_function = new ceres::HuberLoss(imu_huber_delta);
        
        // Add residual block using InertialGravityFactor with shared bias parameters and Huber loss
        // NOTE: InertialGravityFactor expects [pose1, velocity1, GYRO_bias, ACCEL_bias, pose2, velocity2, gravity_dir]
        problem.AddResidualBlock(inertial_gravity_factor, imu_loss_function,
                                const_cast<double*>(pose_params_vec[i].data()),           // pose_i (6D SE3)
                                const_cast<double*>(velocity_params_vec[i].data()),       // velocity_i (3D)
                                const_cast<double*>(gyro_bias_params.data()),             // shared GYRO_bias (3D) - parameters[2]
                                const_cast<double*>(accel_bias_params.data()),            // shared ACCEL_bias (3D) - parameters[3]
                                const_cast<double*>(pose_params_vec[i+1].data()),         // pose_j (6D SE3)
                                const_cast<double*>(velocity_params_vec[i+1].data()),     // velocity_j (3D)
                                const_cast<double*>(gravity_dir_params.data()));          // gravity_dir (2D sphere)
        
        factors_added++;
        
    }
    
    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("[SW_IMU] Added {} InertialGravityFactor factors to sliding window with shared bias", factors_added);
    }
    return factors_added;
}







void SlidingWindowOptimizer::setup_imu_parameter_blocks(
    ceres::Problem& problem,
    const std::vector<std::shared_ptr<Frame>>& keyframes,
    std::vector<std::vector<double>>& velocity_params_vec,
    std::vector<double>& accel_bias_params,
    std::vector<double>& gyro_bias_params,
    std::vector<double>& gravity_dir_params) {
    
    if (!m_imu_enabled) {
        return;
    }
    
    size_t num_keyframes = keyframes.size();
    
    // Initialize velocity parameters for each keyframe (per-frame velocities)
    velocity_params_vec.resize(num_keyframes);
    for (size_t i = 0; i < num_keyframes; ++i) {
        velocity_params_vec[i].resize(3);
        Eigen::Vector3f velocity = keyframes[i]->get_velocity();
        velocity_params_vec[i][0] = velocity.x();
        velocity_params_vec[i][1] = velocity.y(); 
        velocity_params_vec[i][2] = velocity.z();
        
        // Add velocity parameter block
        problem.AddParameterBlock(velocity_params_vec[i].data(), 3);
    }
    
    // Initialize shared accelerometer bias (global parameter)
    accel_bias_params.resize(3);
    if (!keyframes.empty()) {
        Eigen::Vector3f accel_bias = keyframes[0]->get_accel_bias();  // Get from first keyframe
        accel_bias_params[0] = accel_bias.x();
        accel_bias_params[1] = accel_bias.y();
        accel_bias_params[2] = accel_bias.z();
    } else {
        accel_bias_params[0] = 0.0;
        accel_bias_params[1] = 0.0;
        accel_bias_params[2] = 0.0;
    }
    problem.AddParameterBlock(accel_bias_params.data(), 3);
    
    // Initialize shared gyroscope bias (global parameter)
    gyro_bias_params.resize(3);
    if (!keyframes.empty()) {
        Eigen::Vector3f gyro_bias = keyframes[0]->get_gyro_bias();  // Get from first keyframe
        gyro_bias_params[0] = gyro_bias.x();
        gyro_bias_params[1] = gyro_bias.y();
        gyro_bias_params[2] = gyro_bias.z();
    } else {
        gyro_bias_params[0] = 0.0;
        gyro_bias_params[1] = 0.0;
        gyro_bias_params[2] = 0.0;
    }
    problem.AddParameterBlock(gyro_bias_params.data(), 3);
    
    // Initialize gravity direction parameters (fixed)
    gravity_dir_params.resize(2);
    // Convert gravity direction to sphere parameterization (theta, phi)
    gravity_dir_params[0] = 0.0;
    gravity_dir_params[1] = 0.0;
    
    // Add gravity direction parameter block (constant during sliding window)
    problem.AddParameterBlock(gravity_dir_params.data(), 2);
    problem.SetParameterBlockConstant(gravity_dir_params.data()); // Fixed gravity direction
    
    // Add bias priors to prevent drift - bias should stay close to current values
    Eigen::Vector3d accel_bias_prior(accel_bias_params[0], accel_bias_params[1], accel_bias_params[2]);
    Eigen::Vector3d gyro_bias_prior(gyro_bias_params[0], gyro_bias_params[1], gyro_bias_params[2]);
    
    // Compute bias prior weights dynamically from IMU handler covariance
    double accel_bias_weight, gyro_bias_weight;
    
    if (m_imu_handler && !keyframes.empty()) {
        // Try to extract bias uncertainty from latest preintegration covariance
        auto latest_frame = keyframes.back();
        auto preintegration = latest_frame->get_imu_preintegration_from_last_keyframe();
        
        if (preintegration) {
            // Extract bias covariance from 15x15 preintegration covariance matrix
            // Structure: [δR(3), δV(3), δP(3), δbg(3), δba(3)]
            // Gyro bias covariance: indices 9-11, Accel bias covariance: indices 12-14
            Eigen::Matrix3f gyro_bias_cov = preintegration->covariance.block<3,3>(9, 9);
            Eigen::Matrix3f accel_bias_cov = preintegration->covariance.block<3,3>(12, 12);
            
            // // Log covariance diagonal for debugging
            // spdlog::debug("[SW_IMU] Current bias covariances - Gyro diag: ({:.2e}, {:.2e}, {:.2e}), Accel diag: ({:.2e}, {:.2e}, {:.2e})",
            //              gyro_bias_cov(0,0), gyro_bias_cov(1,1), gyro_bias_cov(2,2),
            //              accel_bias_cov(0,0), accel_bias_cov(1,1), accel_bias_cov(2,2));
            
            // Compute weights as inverse of diagonal covariance elements (information matrix)
            // Method 1: Use average variance across all 3 axes
            double gyro_bias_variance = gyro_bias_cov.trace() / 3.0 + 1e-8;  // Average variance
            double accel_bias_variance = accel_bias_cov.trace() / 3.0 + 1e-8;
            
            // Alternative methods (choose one):
            // Method 2: Use maximum variance (most conservative)
            // double gyro_bias_variance = std::max({gyro_bias_cov(0,0), gyro_bias_cov(1,1), gyro_bias_cov(2,2)}) + 1e-8;
            // double accel_bias_variance = std::max({accel_bias_cov(0,0), accel_bias_cov(1,1), accel_bias_cov(2,2)}) + 1e-8;
            
            // Method 3: Use determinant-based measure
            // double gyro_bias_variance = std::pow(gyro_bias_cov.determinant(), 1.0/3.0) + 1e-8;  // Geometric mean
            // double accel_bias_variance = std::pow(accel_bias_cov.determinant(), 1.0/3.0) + 1e-8;
            
            gyro_bias_weight = 1.0 / gyro_bias_variance;
            accel_bias_weight = 1.0 / accel_bias_variance;

            // std::cout<<gyro_bias_weight<< " / "<<   accel_bias_weight<<std::endl;
            
            // // Apply reasonable bounds to prevent extreme weights
            gyro_bias_weight = std::clamp(gyro_bias_weight, 1.0, 1e5);
            accel_bias_weight = std::clamp(accel_bias_weight, 1.0, 1e4);
            
            // spdlog::debug("[SW_IMU] Dynamic bias weights from covariance: gyro={:.2e}, accel={:.2e}", 
            //              gyro_bias_weight, accel_bias_weight);
        } else {
            // Fallback to default values if no preintegration available
            accel_bias_weight = 1e4;
            gyro_bias_weight = 1e5;
            spdlog::debug("[SW_IMU] Using fallback bias weights (no preintegration): gyro={:.2e}, accel={:.2e}", 
                         gyro_bias_weight, accel_bias_weight);
        }
    } else {
        // Fallback to default values if IMU handler not available
        accel_bias_weight = 1e4;
        gyro_bias_weight = 1e5;
        spdlog::debug("[SW_IMU] Using default bias weights (no IMU handler): gyro={:.2e}, accel={:.2e}", 
                     gyro_bias_weight, accel_bias_weight);
    }
    
    Eigen::Matrix3d accel_bias_info = Eigen::Matrix3d::Identity() * accel_bias_weight;
    Eigen::Matrix3d gyro_bias_info = Eigen::Matrix3d::Identity() * gyro_bias_weight;
    
    // Create and add bias prior cost functions
    auto* accel_bias_prior_cost = new factor::VectorPriorFactor<3>(accel_bias_prior, accel_bias_info);
    auto* gyro_bias_prior_cost = new factor::VectorPriorFactor<3>(gyro_bias_prior, gyro_bias_info);
    
    problem.AddResidualBlock(accel_bias_prior_cost, nullptr, accel_bias_params.data());
    problem.AddResidualBlock(gyro_bias_prior_cost, nullptr, gyro_bias_params.data());
    
    // spdlog::debug("[SW_IMU] Shared accel bias: ({:.10f}, {:.10f}, {:.10f})", accel_bias_params[0], accel_bias_params[1], accel_bias_params[2]);
    // spdlog::debug("[SW_IMU] Shared gyro bias: ({:.10f}, {:.10f}, {:.10f})", gyro_bias_params[0], gyro_bias_params[1], gyro_bias_params[2]);
}

void SlidingWindowOptimizer::update_imu_optimized_values(
    const std::vector<std::shared_ptr<Frame>>& keyframes,
    const std::vector<std::vector<double>>& velocity_params_vec,
    const std::vector<double>& accel_bias_params,
    const std::vector<double>& gyro_bias_params) {
    
    if (!m_imu_enabled) {
        return;
    }
    
    // Update velocities for each keyframe (per-frame velocities)
    for (size_t i = 0; i < keyframes.size(); ++i) {
        auto frame = keyframes[i];
        
        // Update velocity
        Eigen::Vector3f optimized_velocity(
            velocity_params_vec[i][0],
            velocity_params_vec[i][1],
            velocity_params_vec[i][2]
        );
        frame->set_velocity(optimized_velocity);
        
    }
    
    // Update shared bias for all keyframes (same bias applied to all)
    Eigen::Vector3f optimized_accel_bias(
        accel_bias_params[0],
        accel_bias_params[1], 
        accel_bias_params[2]
    );
    
    Eigen::Vector3f optimized_gyro_bias(
        gyro_bias_params[0],
        gyro_bias_params[1],
        gyro_bias_params[2]
    );
    
    // Get initial bias from first frame for comparison
    Eigen::Vector3f initial_accel_bias = keyframes[0]->get_accel_bias();
    Eigen::Vector3f initial_gyro_bias = keyframes[0]->get_gyro_bias();
    
    // Calculate bias change
    Eigen::Vector3f accel_bias_change = optimized_accel_bias - initial_accel_bias;
    Eigen::Vector3f gyro_bias_change = optimized_gyro_bias - initial_gyro_bias;
    
    // Apply shared bias to all keyframes
    for (size_t i = 0; i < keyframes.size(); ++i) {
        auto& frame = keyframes[i];
        frame->set_accel_bias(optimized_accel_bias);
        frame->set_gyro_bias(optimized_gyro_bias);
        
       
    }
    
    // Update IMU handler's global bias if available
    if (m_imu_handler) {
        // Get current IMU handler bias for comparison
        Eigen::Vector3f handler_gyro_bias = m_imu_handler->get_gyro_bias();
        Eigen::Vector3f handler_accel_bias = m_imu_handler->get_accel_bias();
        
        m_imu_handler->set_bias(optimized_gyro_bias, optimized_accel_bias);
        
        // Update all preintegrations with optimized bias (like in IMU initialization)
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> frame_biases;
        for (size_t i = 1; i < keyframes.size(); ++i) { // Skip first frame
            frame_biases.emplace_back(optimized_gyro_bias, optimized_accel_bias); // Shared bias for all
        }
        
        // Convert shared_ptr<Frame> to Frame* for IMU handler call
        std::vector<Frame*> raw_frames_for_update;
        for (size_t i = 1; i < keyframes.size(); ++i) { // Skip first frame
            raw_frames_for_update.push_back(keyframes[i].get());
        }
        
        m_imu_handler->update_preintegrations_with_optimized_bias(raw_frames_for_update, frame_biases);
        
    }
    
}

Eigen::Matrix2d PnPOptimizer::create_information_from_uncertainty_propagation(
    std::shared_ptr<MapPoint> mappoint,
    std::shared_ptr<Frame> frame) const
{

    // Get world uncertainty from MapPoint (3x3 covariance matrix)
    Eigen::Matrix3d world_uncertainty = mappoint->get_world_uncertainty().cast<double>();

    Eigen::Matrix2d information_matrix = mappoint->transform_uncertainty_world_to_pixel(world_uncertainty.cast<float>(), frame).cast<double>().inverse();

    return information_matrix + Eigen::Matrix2d::Identity(); // Add small value to diagonal for numerical stability
}

} // namespace lightweight_vio
