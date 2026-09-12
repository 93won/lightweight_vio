/**
 * @file      Optimizer.h
 * @brief     Defines pose and bundle adjustment optimizers using Ceres Solver.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <memory>
#include <mutex>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

// Forward declarations
namespace lightweight_vio {
    class Frame;
    class Feature;
    class MapPoint;
    class IMUHandler;
    struct IMUPreintegration;
    
    namespace factor {
        class PnPFactor;
        class BAFactor;
        class IMUFactor;
        struct CameraParameters;
    }
}

namespace lightweight_vio {

/**
 * @brief Pose optimization result
 */
struct OptimizationResult {
    bool success;
    int num_inliers;
    int num_outliers;
    Eigen::Matrix4f optimized_pose;
    std::vector<bool> outlier_flags;
    double initial_cost;
    double final_cost;
    int num_iterations;
    
    OptimizationResult() : success(false), num_inliers(0), num_outliers(0), 
                          initial_cost(0.0), final_cost(0.0), num_iterations(0) {}
};

/**
 * @brief Inertial optimization result
 */
struct InertialOptimizationResult {
    bool success;
    int num_iterations;
    double initial_cost;
    double final_cost;
    double cost_reduction;
    
    // Detailed cost breakdown
    double visual_cost;
    double imu_cost;
    double bias_cost;
    
    // Optimization statistics
    int num_visual_residuals;
    int num_imu_residuals;
    int num_outliers_removed;
    
    // Gravity transformation matrix (World-to-Gravity, SE(3))
    Eigen::Matrix4f Tgw_init;
    Eigen::Matrix3d Rwg;  // Rotation from world to gravity frame
    
    // ⭐ Gravity vector in World frame (BEFORE transformation) for visualization
    Eigen::Vector3f g_world_before_transform;
    Eigen::Vector3f first_frame_position;  // Position to draw arrow from
    bool has_gravity_visualization_data;
    
    // Optimized states
    std::vector<Eigen::Vector3f> optimized_velocities;
    Eigen::Vector3f optimized_gyro_bias;
    Eigen::Vector3f optimized_accel_bias;
    
    InertialOptimizationResult() 
        : success(false), num_iterations(0), initial_cost(0.0), final_cost(0.0),
          cost_reduction(0.0), visual_cost(0.0), imu_cost(0.0), bias_cost(0.0),
          num_visual_residuals(0), num_imu_residuals(0), num_outliers_removed(0),
          Tgw_init(Eigen::Matrix4f::Identity()),
          Rwg(Eigen::Matrix3d::Identity()),
          g_world_before_transform(Eigen::Vector3f::Zero()),
          first_frame_position(Eigen::Vector3f::Zero()),
          has_gravity_visualization_data(false),
          optimized_gyro_bias(Eigen::Vector3f::Zero()),
          optimized_accel_bias(Eigen::Vector3f::Zero()) {}
};

/**
 * @brief Structure to hold both residual block ID and cost function
 */
struct ObservationInfo {
    ceres::ResidualBlockId residual_id;
    factor::PnPFactor* cost_function;
    
    ObservationInfo(ceres::ResidualBlockId id, factor::PnPFactor* func)
        : residual_id(id), cost_function(func) {}
};

/**
 * @brief Pose optimizer for VIO system using Ceres
 */
class PnPOptimizer {
public:
    /**
     * @brief Constructor
     */
    PnPOptimizer();
    
    /**
     * @brief Destructor
     */
    ~PnPOptimizer() = default;
    
    /**
     * @brief Optimize pose using PnP with map points
     * @param frame Frame to optimize pose for
     * @return Optimization result
     */
    OptimizationResult optimize_pose(std::shared_ptr<Frame> frame);

private:

    /**
     * @brief Add a mono observation residual with adaptive weighting based on config mode
     * @param problem Ceres problem instance
     * @param pose_params Pose parameters (6-DOF SE3)
     * @param world_point 3D world point
     * @param observation 2D image observation
     * @param camera_params Camera intrinsics
     * @param mappoint MapPoint for adaptive uncertainty weighting (optional)
     * @param frame Frame containing the observation
     * @param pixel_noise_std Standard deviation of pixel noise
     * @return Observation info with residual block ID and cost function
     */
    ObservationInfo add_observation(
        ceres::Problem& problem,
        double* pose_params,
        const Eigen::Vector3d& world_point,
        const Eigen::Vector2d& observation,
        const factor::CameraParameters& camera_params,
        std::shared_ptr<MapPoint> mappoint = nullptr,
        std::shared_ptr<Frame> frame = nullptr,
        const double pixel_noise_std = 1.0
        );
    
    /**
     * @brief Detect outliers using chi-square test
     * @param pose_params Current pose parameters
     * @param observations Vector of observation info
     * @param feature_indices Indices of features corresponding to observations
     * @param frame Frame to update outlier flags
     * @return Number of inliers
     */
    int detect_outliers(double const* const* pose_params,
                       const std::vector<ObservationInfo>& observations,
                       const std::vector<int>& feature_indices,
                       std::shared_ptr<Frame> frame);
    
    /**
     * @brief Setup solver options based on configuration
     * @return Configured solver options
     */
    ceres::Solver::Options setup_solver_options(int max_iter) const;
    
    /**
     * @brief Convert frame pose to SE3 tangent space
     * @param frame Frame with pose to convert
     * @return SE3 tangent space vector [6]
     */
    Sophus::SE3d::Tangent frame_to_se3_tangent(std::shared_ptr<Frame> frame) const;
    
    /**
     * @brief Convert SE3 tangent space to 4x4 matrix
     * @param se3_tangent SE3 tangent space vector [6]
     * @return 4x4 transformation matrix
     */
    Eigen::Matrix4f se3_tangent_to_matrix(const Sophus::SE3d::Tangent& se3_tangent) const;
    
    /**
     * @brief Create robust loss function
     * @param delta Huber loss delta parameter
     * @return Robust loss function
     */
    ceres::LossFunction* create_robust_loss(double delta) const;
    
    /**
     * @brief Create information matrix for pixel observations
     * @param pixel_noise Standard deviation of pixel noise
     * @return 2x2 information matrix
     */
    Eigen::Matrix2d create_information_matrix(double pixel_noise = 1.0) const;
    
    /**
     * @brief Create adaptive information matrix based on MapPoint uncertainty when enabled
     * @param mappoint MapPoint with world uncertainty information
     * @param frame Frame for coordinate transformations
     * @param camera_params Camera intrinsics for projection
     * @param pixel_noise_std Standard deviation of pixel noise
     * @return 2x2 information matrix (inverse of covariance)
     */
    Eigen::Matrix2d create_information_from_uncertainty_propagation(std::shared_ptr<MapPoint> mappoint,
                                                       std::shared_ptr<Frame> frame) const;



private:
    // Global mutexes for thread-safe access to MapPoints and Frames
    static std::mutex s_mappoint_mutex;
    static std::mutex s_keyframe_mutex;


    // Debugging
    std::vector<double> m_pnp_info_x_sqrt, m_pnp_info_y_sqrt;
};

/**
 * @brief Sliding Window Bundle Adjustment result
 */
struct SlidingWindowResult {
    bool success;
    int num_inliers;
    int num_outliers;
    int num_poses_optimized;
    int num_points_optimized;
    double initial_cost;
    double final_cost;
    int num_iterations;
    
    SlidingWindowResult() : success(false), num_inliers(0), num_outliers(0),
                           num_poses_optimized(0), num_points_optimized(0),
                           initial_cost(0.0), final_cost(0.0), num_iterations(0) {}
};

/**
 * @brief Structure to hold bundle adjustment observation info
 */
struct BAObservationInfo {
    ceres::ResidualBlockId residual_id;
    factor::BAFactor* cost_function;
    int keyframe_index;  // Index in sliding window
    int mappoint_index;  // Index in map points vector
    bool is_right_camera = false;

    Eigen::Matrix2d information_matrix; // Information matrix for this observation

    BAObservationInfo(ceres::ResidualBlockId id, factor::BAFactor* func,
                     int kf_idx, int mp_idx, const Eigen::Matrix2d& info_matrix)
        : residual_id(id), cost_function(func),
          keyframe_index(kf_idx), mappoint_index(mp_idx),
          information_matrix(info_matrix) {}
};

/**
 * @brief Sliding Window Bundle Optimizer for VIO system using Ceres
 */
class SlidingWindowOptimizer {
public:
    /**
     * @brief Constructor
     * @param window_size Maximum number of keyframes in sliding window
     */
    SlidingWindowOptimizer(size_t window_size = 10);
    
    /**
     * @brief Destructor
     */
    ~SlidingWindowOptimizer() = default;
    
    /**
     * @brief Optimize sliding window of keyframes and map points
     * @param keyframes Vector of keyframes in sliding window (oldest to newest)
     * @return Optimization result
     */
    SlidingWindowResult optimize(const std::vector<std::shared_ptr<Frame>>& keyframes);
    
    /**
     * @brief Enable IMU optimization in sliding window
     * @param imu_handler IMU handler with preintegration data
     * @param gravity_direction Fixed gravity direction (normalized)
     */
    void enable_imu_optimization(std::shared_ptr<IMUHandler> imu_handler,
                                const Eigen::Vector3d& gravity_direction);
    
    /**
     * @brief Disable IMU optimization (visual-only mode)
     */
    void disable_imu_optimization();
    
    /**
     * @brief Check if IMU optimization is enabled
     * @return True if IMU optimization is enabled
     */
    bool is_imu_enabled() const { return m_imu_enabled; }

private:
    /**
     * @brief Collect map points observed by keyframes in sliding window
     * @param keyframes Vector of keyframes
     * @return Vector of unique map points
     */
    std::vector<std::shared_ptr<MapPoint>> collect_window_map_points(
        const std::vector<std::shared_ptr<Frame>>& keyframes) const;
    
    /**
     * @brief Add bundle adjustment observation to problem
     * @param problem Ceres problem
     * @param pose_params Pose parameters for keyframe
     * @param point_params 3D point parameters
     * @param observation 2D observation
     * @param camera_params Camera parameters
     * @param frame Frame containing the T_CB transformation
     * @param kf_index Keyframe index in sliding window
     * @param mp_index Map point index
     * @return Observation info with residual block ID and cost function
     */
    BAObservationInfo add_observation(
        ceres::Problem& problem,
        double* pose_params,
        double* point_params,
        const Eigen::Vector2d& observation,
        const factor::CameraParameters& camera_params,
        std::shared_ptr<Frame> frame,
        std::shared_ptr<MapPoint> mappoint,
        int kf_index,
        int mp_index,
        double pixel_noise_std = 1.0);
    
   
    
    /**
     * @brief Setup optimization problem with keyframes and map points
     * @param problem Ceres problem
     * @param keyframes Vector of keyframes
     * @param map_points Vector of map points
     * @param pose_params_vec Vector of pose parameter arrays
     * @param point_params_vec Vector of point parameter arrays
     * @return Vector of observation info
     */
    std::vector<BAObservationInfo> setup_optimization_problem(
        ceres::Problem& problem,
        const std::vector<std::shared_ptr<Frame>>& keyframes,
        const std::vector<std::shared_ptr<MapPoint>>& map_points,
        std::vector<std::vector<double>>& pose_params_vec,
        std::vector<std::vector<double>>& point_params_vec);
    
    /**
     * @brief Apply marginalization strategy
     * @param problem Ceres problem
     * @param keyframes Vector of keyframes
     * @param map_points Vector of map points
     * @param pose_params_vec Vector of pose parameter arrays
     * @param point_params_vec Vector of point parameter arrays
     * @param num_fixed_keyframes Number of keyframes to fix from the beginning (default: 1)
     * @param reset_constraints Whether to reset existing constraints before applying new ones (default: false)
     */
    void apply_marginalization_strategy(
        ceres::Problem& problem,
        const std::vector<std::shared_ptr<Frame>>& keyframes,
        const std::vector<std::shared_ptr<MapPoint>>& map_points,
        const std::vector<std::vector<double>>& pose_params_vec,
        const std::vector<std::vector<double>>& point_params_vec,
        int num_fixed_keyframes = 1,
        bool reset_constraints = false,
        bool marginalize_points = false);
    
    /**
     * @brief Detect outliers using chi-square test for bundle adjustment
     * @param pose_params_vec Vector of pose parameter arrays
     * @param point_params_vec Vector of point parameter arrays
     * @param observations Vector of observation info
     * @param keyframes Vector of keyframes to update outlier flags
     * @return Number of inliers
     */
    int detect_ba_outliers(
        const std::vector<std::vector<double>>& pose_params_vec,
        const std::vector<std::vector<double>>& point_params_vec,
        const std::vector<BAObservationInfo>& observations,
        const std::vector<std::shared_ptr<Frame>>& keyframes,
        const std::vector<std::shared_ptr<MapPoint>>& map_points);
    
    /**
     * @brief Update keyframes and map points with optimized values
     * @param keyframes Vector of keyframes to update
     * @param map_points Vector of map points to update
     * @param pose_params_vec Vector of optimized pose parameters
     * @param point_params_vec Vector of optimized point parameters
     */
    void update_optimized_values(
        const std::vector<std::shared_ptr<Frame>>& keyframes,
        const std::vector<std::shared_ptr<MapPoint>>& map_points,
        const std::vector<std::vector<double>>& pose_params_vec,
        const std::vector<std::vector<double>>& point_params_vec);
    
    /**
     * @brief Setup solver options for sliding window optimization
     * @return Configured solver options
     */
    ceres::Solver::Options setup_solver_options(int max_iter) const;
    
    /**
     * @brief Create robust loss function
     * @param delta Huber loss delta parameter
     * @return Robust loss function
     */
    ceres::LossFunction* create_robust_loss(double delta) const;
    
    /**
     * @brief Create information matrix for pixel observations
     * @param pixel_noise Standard deviation of pixel noise
     * @return 2x2 information matrix
     */
    Eigen::Matrix2d create_information_matrix(double pixel_noise = 1.0) const;
    
    /**
     * @brief Create information matrix with observation-based weighting
     * @param pixel_noise Standard deviation of pixel noise
     * @param num_observations Number of observations for weighting (max 3.0)
     * @return 2x2 information matrix
     */
    Eigen::Matrix2d create_information_matrix_with_num_observations(double pixel_noise, int num_observations) const;
    
    /**
     * @brief Create adaptive information matrix based on MapPoint uncertainty when enabled
     * @param mappoint MapPoint with world uncertainty information
     * @param frame Frame for coordinate transformations
     * @param camera_params Camera intrinsics for projection
     * @param pixel_noise_std Standard deviation of pixel noise
     * @return 2x2 information matrix (inverse of covariance)
     */
    Eigen::Matrix2d create_information_from_uncertainty_propagation(std::shared_ptr<MapPoint> mappoint,
                                                       std::shared_ptr<Frame> frame) const;
    
    /**
     * @brief Add inertial factors to sliding window optimization using existing InertialGravityFactor
     * @param problem Ceres problem
     * @param keyframes Vector of keyframes  
     * @param pose_params_vec Vector of pose parameter arrays
     * @param velocity_params_vec Vector of velocity parameter arrays  
     * @param accel_bias_params Shared accelerometer bias parameters (global)
     * @param gyro_bias_params Shared gyroscope bias parameters (global)
     * @param gravity_dir_params Fixed gravity direction parameters
     * @return Number of IMU factors added
     */
    int add_inertial_factors_to_sliding_window(
        ceres::Problem& problem,
        const std::vector<std::shared_ptr<Frame>>& keyframes,
        const std::vector<std::vector<double>>& pose_params_vec,
        const std::vector<std::vector<double>>& velocity_params_vec,
        const std::vector<double>& accel_bias_params,
        const std::vector<double>& gyro_bias_params,
        const std::vector<double>& gravity_dir_params);
    
    /**
     * @brief Setup IMU parameter blocks for sliding window optimization
     * @param problem Ceres problem
     * @param keyframes Vector of keyframes
     * @param velocity_params_vec Vector to store velocity parameters (per keyframe)
     * @param accel_bias_params Shared accelerometer bias parameters (global)
     * @param gyro_bias_params Shared gyroscope bias parameters (global)
     * @param gravity_dir_params Vector to store gravity direction parameters
     */
    void setup_imu_parameter_blocks(
        ceres::Problem& problem,
        const std::vector<std::shared_ptr<Frame>>& keyframes,
        std::vector<std::vector<double>>& velocity_params_vec,
        std::vector<double>& accel_bias_params,
        std::vector<double>& gyro_bias_params,
        std::vector<double>& gravity_dir_params);
        
    /**
     * @brief Update frames with optimized IMU states
     * @param keyframes Vector of keyframes to update
     * @param velocity_params_vec Optimized velocity parameters (per keyframe)
     * @param accel_bias_params Optimized accelerometer bias parameters (shared)
     * @param gyro_bias_params Optimized gyroscope bias parameters (shared)
     */
    void update_imu_optimized_values(
        const std::vector<std::shared_ptr<Frame>>& keyframes,
        const std::vector<std::vector<double>>& velocity_params_vec,
        const std::vector<double>& accel_bias_params,
        const std::vector<double>& gyro_bias_params);

private:
    size_t m_window_size;        // Maximum keyframes in sliding window
    int m_max_iterations;        // Maximum optimization iterations
    double m_huber_delta;        // Huber loss delta parameter
    double m_pixel_noise_std;    // Pixel noise standard deviation
    double m_outlier_threshold;  // Chi-square outlier threshold
    
    // IMU optimization settings
    bool m_imu_enabled;          // Whether IMU optimization is enabled
    std::shared_ptr<IMUHandler> m_imu_handler;  // IMU handler for preintegration
    Eigen::Vector3d m_gravity_direction;         // Fixed gravity direction (normalized)
    double m_gravity_magnitude;   // Gravity magnitude (m/s^2)
    
    // Global mutexes for thread-safe access to MapPoints and Frames
    static std::mutex s_mappoint_mutex;
    static std::mutex s_keyframe_mutex;


    // Debugging
    std::vector<double> m_sba_info_x_sqrt, m_sba_info_y_sqrt;

};

/**
 * @brief Inertial optimizer for VIO system
 */
class InertialOptimizer {
public:
    /**
     * @brief Constructor
     */
    InertialOptimizer();
    
    /**
     * @brief Destructor
     */
    ~InertialOptimizer() = default;
    
    /**
     * @brief IMU initialization optimization using InertialGravityFactor
     * @param frames Frames to optimize (poses fixed, velocities and biases optimized)
     * @param all_keyframes_for_transform All keyframes to transform to gravity-aligned frame
     * @param imu_handler IMU handler with preintegration data
     * @return Optimization result
     */
    InertialOptimizationResult optimize_imu_initialization(
        std::vector<Frame*>& frames,
        const std::vector<Frame*>& all_keyframes_for_transform,
        std::shared_ptr<IMUHandler> imu_handler);
    
 
private:
    struct OptimizationParams {
        int max_iterations = 200;
        double function_tolerance = 1e-6;
        double gradient_tolerance = 1e-10;
        double parameter_tolerance = 1e-8;
        bool use_robust_kernel = true;
        double huber_delta = 1.0;
    } m_params;
    
    // Helper functions
    // IMU initialization helper methods
    void setup_imu_init_vertices(
        const std::vector<Frame*>& frames,
        std::shared_ptr<IMUHandler> imu_handler,
        std::vector<std::vector<double>>& pose_params_vec,
        std::vector<std::vector<double>>& velocity_params_vec,
        std::vector<std::vector<double>>& accel_bias_params_vec,
        std::vector<std::vector<double>>& gyro_bias_params_vec,
        std::vector<double>& gravity_dir_params);
    
    int add_inertial_gravity_factors(
        ceres::Problem& problem,
        const std::vector<Frame*>& frames,
        std::shared_ptr<IMUHandler> imu_handler,
        const std::vector<std::vector<double>>& pose_params_vec,
        const std::vector<std::vector<double>>& velocity_params_vec,
        const std::vector<std::vector<double>>& accel_bias_params_vec,
        const std::vector<std::vector<double>>& gyro_bias_params_vec,
        const std::vector<double>& gravity_dir_params);
    
    void add_imu_init_priors(
        ceres::Problem& problem,
        const std::vector<Frame*>& frames,
        const std::vector<std::vector<double>>& velocity_params_vec,
        const std::vector<std::vector<double>>& accel_bias_params_vec,
        const std::vector<std::vector<double>>& gyro_bias_params_vec);
    
    void recover_imu_init_states(
        const std::vector<Frame*>& frames,
        const std::vector<Frame*>& all_frames_for_transform,
        std::shared_ptr<IMUHandler> imu_handler,
        const std::vector<std::vector<double>>& pose_params_vec,
        const std::vector<std::vector<double>>& velocity_params_vec,
        const std::vector<std::vector<double>>& accel_bias_params_vec,
        const std::vector<std::vector<double>>& gyro_bias_params_vec,
        const std::vector<double>& gravity_dir_params,
        Eigen::Matrix4f& T_gw);
    
    // Optimization setup and execution methods
    void setup_params(
        const std::vector<Frame*>& frames,
        std::vector<std::vector<double>>& pose_params_vec,
        std::vector<std::vector<double>>& velocity_params_vec,
        std::vector<double>& gyro_bias_params,
        std::vector<double>& accel_bias_params);

    
    int add_inertial_factors(
        ceres::Problem& problem,
        const std::vector<Frame*>& frames,
        std::shared_ptr<IMUHandler> imu_handler,
        const Eigen::Vector3d& gravity,
        const std::vector<std::vector<double>>& pose_params_vec,
        const std::vector<std::vector<double>>& velocity_params_vec,
        double* gyro_bias_params,
        double* accel_bias_params);
    
    void recover_optimized_states(
        const std::vector<Frame*>& frames,
        const std::vector<std::vector<double>>& pose_params_vec,
        const std::vector<std::vector<double>>& velocity_params_vec,
        const double* gyro_bias_params,
        const double* accel_bias_params);
    
    void cleanup_vertices(
        std::vector<std::vector<double>>& pose_params_vec,
        std::vector<std::vector<double>>& velocity_params_vec);
};


} // namespace lightweight_vio
