/**
 * @file      Estimator.h
 * @brief     Defines the main VO estimation class.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-23
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
// Forward declarations


namespace lightweight_vio {
    class Frame;
    class MapPoint;
    class FeatureTracker;
    class IMUHandler;
    class InertialOptimizer;
    class PnPOptimizer;
    class SlidingWindowOptimizer;

    struct OptimizationResult;
    struct IMUData;
    struct IMUPreintegration;
}


namespace lightweight_vio {

/**
 * @brief Main VIO estimator class that handles the complete pipeline
 */
class Estimator {
public:
    /**
     * @brief VIO estimation result
     */
    struct EstimationResult {
        bool success;
        Eigen::Matrix4f pose;
        int num_features;
        int num_inliers;
        int num_outliers;
        int num_new_map_points;
        int num_tracked_features;
        int num_features_with_map_points;
        double optimization_time_ms;
        
        EstimationResult() : success(false), pose(Eigen::Matrix4f::Identity()),
                           num_features(0), num_inliers(0), num_outliers(0),
                           num_new_map_points(0), num_tracked_features(0), 
                           num_features_with_map_points(0), optimization_time_ms(0.0) {}
    };

    /**
     * @brief Constructor
     */
    Estimator();
    
    /**
     * @brief Destructor
     */
    ~Estimator();

    /**
     * @brief Process a new stereo frame
     * @param left_image Left stereo image
     * @param right_image Right stereo image  
     * @param timestamp Frame timestamp in nanoseconds
     * @return Estimation result
     */
    EstimationResult process_frame(const cv::Mat& left_image, const cv::Mat& right_image, long long timestamp);

    /**
     * @brief Process a new stereo frame with IMU data
     * @param left_image Left stereo image
     * @param right_image Right stereo image  
     * @param timestamp Frame timestamp in nanoseconds
     * @param imu_data_from_last_frame IMU measurements between last frame and current frame
     * @return Estimation result
     */
    EstimationResult process_frame(const cv::Mat& left_image, const cv::Mat& right_image, 
                                 long long timestamp, const std::vector<IMUData>& imu_data_from_last_frame);

    /**
     * @brief Reset the estimator state
     */
    void reset();

    /**
     * @brief Get current pose
     * @return Current camera pose (Twb)
     */
    Eigen::Matrix4f get_current_pose() const;

    /**
     * @brief Set initial ground truth pose for first frame
     * @param gt_pose Ground truth pose (Twb)
     */
    void set_initial_gt_pose(const Eigen::Matrix4f& gt_pose);
    
    /**
     * @brief Apply ground truth pose to current frame (for debugging)
     * @param gt_pose Ground truth pose (Twb)
     */
    void apply_gt_pose_to_current_frame(const Eigen::Matrix4f& gt_pose);

    /**
     * @brief Get all keyframes (not thread-safe)
     * @return Vector of keyframes
     */
    const std::vector<std::shared_ptr<Frame>>& get_keyframes() const { return m_keyframes; }
    

   
    /**
     * @brief Get all keyframes (thread-safe copy)
     * @return Copy of keyframes vector
     */
    std::vector<std::shared_ptr<Frame>> get_keyframes_safe() const;

    /**
     * @brief Get all processed frames (for trajectory export)
     * @return Vector of all processed frames
     */
    const std::vector<std::shared_ptr<Frame>>& get_all_frames() const { return m_all_frames; }

    /**
     * @brief Get all map points
     * @return Vector of map points
     */
    const std::vector<std::shared_ptr<MapPoint>>& get_map_points() const { return m_map_points; }

    /**
     * @brief Get all map points (thread-safe copy)
     * @return Copy of map points vector
     */
    std::vector<std::shared_ptr<MapPoint>> get_map_points_safe() const;

    /**
     * @brief Get current frame
     * @return Current frame (can be nullptr)
     */
    std::shared_ptr<Frame> get_current_frame() const { return m_current_frame; }

    /**
     * @brief Get the World-to-Gravity transformation matrix for initial setup
     * @return 4x4 SE(3) transformation matrix (Identity if not initialized)
     */
    Eigen::Matrix4f get_Tgw_init() const { return m_Tgw_init; }

    /**
     * @brief Check if gravity has been initialized
     * @return True if gravity initialization is complete
     */
    bool is_gravity_initialized() const { return m_gravity_initialized; }

private:
    // System components
    std::unique_ptr<FeatureTracker> m_feature_tracker;
    std::unique_ptr<PnPOptimizer> m_pose_optimizer;
    std::unique_ptr<SlidingWindowOptimizer> m_sliding_window_optimizer;
    std::unique_ptr<IMUHandler> m_imu_handler;  // IMU processing and preintegration
    std::unique_ptr<InertialOptimizer> m_inertial_optimizer;  // VIO optimization
    
    // State
    std::shared_ptr<Frame> m_current_frame;
    std::shared_ptr<Frame> m_previous_frame;
    std::shared_ptr<Frame> m_last_keyframe;  // Track the last keyframe separately
    std::vector<std::shared_ptr<Frame>> m_keyframes;
    std::vector<std::shared_ptr<Frame>> m_all_frames;  // All processed frames for trajectory export
    std::vector<std::shared_ptr<MapPoint>> m_map_points;
    
    // Frame management
    int m_frame_id_counter;
    int m_frames_since_last_keyframe;
    
    // IMU data management
    std::vector<IMUData> m_imu_vec_from_last_keyframe;  // Accumulate IMU data between keyframes
    
    // IMU initialization state
    bool m_gravity_initialized = false;  // Flag indicating if gravity has been estimated
    int m_frame_count_since_start = 0;   // Counter for frames processed since start
    
    // Keyframe management state
    double m_last_keyframe_grid_coverage = 0.0;  // Grid coverage of the last keyframe
    
    // Current pose
    Eigen::Matrix4f m_current_pose;
    
    // Predicted pose for comparison logging
    Eigen::Matrix4f m_predicted_pose;
    
    // Constant velocity model for pose prediction
    Eigen::Matrix4f m_transform_from_last;  // Transformation from last frame to current frame
    
    // Ground truth initialization
    bool m_has_initial_gt_pose;
    Eigen::Matrix4f m_initial_gt_pose;
    
    // Gravity transformation matrix for viewer
    Eigen::Matrix4f m_Tgw_init = Eigen::Matrix4f::Identity();  // World-to-Gravity transformation matrix (Identity if not initialized)
    Eigen::Matrix3f m_Rgw_init = Eigen::Matrix3f::Identity();  // Gravity rotation matrix (Identity if not initialized)

    // Sliding window optimization thread
    std::unique_ptr<std::thread> m_sliding_window_thread;
    std::atomic<bool> m_sliding_window_thread_running;
    mutable std::mutex m_keyframes_mutex;
    mutable std::mutex m_map_points_mutex;
    std::condition_variable m_keyframes_cv;
    std::atomic<bool> m_keyframes_updated;


    bool m_success_imu_init = false;
    
    // IMU optimization state
    bool m_enable_imu_optimization = false;  // Flag to enable/disable IMU optimization
    Eigen::Vector3f m_accel_bias = Eigen::Vector3f::Zero();  // Current accelerometer bias
    Eigen::Vector3f m_gyro_bias = Eigen::Vector3f::Zero();   // Current gyroscope bias
    
    /**
     * @brief Sliding window optimization thread function
     */
    void sliding_window_thread_function();
    
    /**
     * @brief Initialize a new stereo frame
     * @param left_image Left stereo image
     * @param right_image Right stereo image
     * @param timestamp Frame timestamp
     * @return New frame
     */
    std::shared_ptr<Frame> create_frame(const cv::Mat& left_image, const cv::Mat& right_image, long long timestamp);
    
    /**
     * @brief Predict current frame pose using motion model
     */
    void predict_state();
    
    /**
     * @brief Update transform from last frame for velocity estimation
     */
    void update_transform_from_last();
    
    /**
     * @brief Create initial map points from stereo or motion
     * @param frame Frame to create map points for
     * @return Number of created map points
     */
    int create_initial_map_points(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Create new map points for untracked stereo features
     * @param frame Frame to create map points for
     * @return Number of created map points
     */
    int create_new_map_points(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Count how many features have associated map points
     * @param frame Frame to count
     * @return Number of features with map points
     */
    int count_features_with_map_points(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Decide whether to create a new keyframe
     * @param frame Current frame
     * @return True if should create keyframe
     */
    bool should_create_keyframe(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Calculate grid coverage ratio with features that have map points
     * @param frame Frame to analyze
     * @return Ratio of grid cells that have features with map points (0.0 to 1.0)
     */
    double calculate_grid_coverage_with_map_points(std::shared_ptr<Frame> frame);
    
    
    /**
     * @brief Create a new keyframe
     * @param frame Frame to convert to keyframe
     */
    void create_keyframe(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Transfer accumulated IMU data to keyframe and clear buffer
     * @param keyframe Keyframe to receive IMU data
     */
    void transfer_imu_data_to_keyframe(std::shared_ptr<Frame> keyframe);
    
    /**
     * @brief Check if IMU initialization is possible
     * @return True if conditions are met for IMU initialization
     */
    bool can_initialize_imu() const;
    
    /**
     * @brief Initialize IMU system (bias, gravity, scale)
     * @return True if initialization successful
     */
    bool initialize_imu_system();
    
    /**
     * @brief Attempt IMU initialization with gravity estimation and bias optimization
     * @return True if IMU initialization successful
     */
    bool try_initialize_imu();
    
   
    /**
     * @brief Estimate initial IMU bias from static period
     * @param static_imu_data Static IMU measurements
     */
    void estimate_initial_imu_bias(const std::vector<IMUData>& static_imu_data);
    
    /**
     * @brief Notify sliding window thread about keyframe updates
     */
    void notify_sliding_window_thread();
    
    /**
     * @brief Debug keyframe-to-keyframe IMU vs VO pose differences (after IMU init)
     * @param current_kf Current keyframe to compare with previous keyframe
     */
    void debug_keyframe_to_keyframe_comparison();
    
    /**
     * @brief Perform pose optimization
     * @param frame Frame to optimize pose for
     * @return Optimization result
     */
    OptimizationResult optimize_pose(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Compute reprojection error statistics for map points
     * @param frame Frame to compute reprojection errors for
     */
    void compute_reprojection_error_statistics(std::shared_ptr<Frame> frame);
};

} // namespace lightweight_vio
