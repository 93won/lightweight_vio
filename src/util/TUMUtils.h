/**
 * @file      TUMUtils.h
 * @brief     Defines utilities for handling the TUM dataset.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-09-27
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */


#pragma once

#include <vector>
#include <string>
#include <optional>
#include <Eigen/Dense>

namespace lightweight_vio {

// Forward declaration
struct IMUData;

/**
 * @brief Utility class for TUM dataset specific operations
 * Handles ground truth data loading and timestamp synchronization
 */
class TUMUtils {
public:
    struct GroundTruthPose {
        long long timestamp;
        Eigen::Matrix4f pose;
        Eigen::Vector3f velocity;
        Eigen::Vector3f bias_gyro;
        Eigen::Vector3f bias_accel;
    };

    /**
     * @brief Load ground truth data from TUM dataset
     * @param dataset_root_path Root path of the TUM dataset (e.g., /path/to/V1_01_easy)
     * @return True if loaded successfully
     */
    static bool load_ground_truth(const std::string& dataset_root_path);

    /**
     * @brief Get ground truth pose for a given timestamp
     * @param timestamp Timestamp in nanoseconds
     * @return Ground truth pose matrix (4x4)
     */
    static Eigen::Matrix4f get_ground_truth_pose(long long timestamp);

    /**
     * @brief Get ground truth pose for a given timestamp in seconds
     * @param timestamp_sec Timestamp in seconds
     * @return Ground truth pose matrix (4x4)
     */
    static Eigen::Matrix4f get_ground_truth_pose_sec(double timestamp_sec);

    /**
     * @brief Pre-match ground truth poses for given image timestamps
     * @param image_timestamps Vector of image timestamps in nanoseconds
     * @return True if matching was successful
     */
    static bool match_image_timestamps(const std::vector<long long>& image_timestamps);

    /**
     * @brief Get pre-matched ground truth pose by image index
     * @param image_index Index of the image (0-based)
     * @return Ground truth pose matrix (4x4)
     */
    static std::optional<Eigen::Matrix4f> get_matched_pose(size_t image_index);

    /**
     * @brief Get number of matched poses
     * @return Number of pre-matched poses
     */
    static size_t get_matched_count();

    /**
     * @brief Get matched image timestamp at given index
     * @param index Index of the matched timestamp
     * @return Timestamp in nanoseconds, or 0 if invalid index
     */
    static long long get_matched_timestamp(size_t index);

    /**
     * @brief Print statistics about loaded ground truth data
     */
    static void print_ground_truth_stats();

    /**
     * @brief Check if ground truth data is loaded
     * @return True if data is available
     */
    static bool has_ground_truth();



    /**
     * @brief Load IMU data from TUM dataset
     * @param dataset_root_path Root path of the TUM dataset (e.g., /path/to/V1_01_easy)
     * @return True if loaded successfully
     */
    static bool load_imu_data(const std::string& dataset_root_path);

    /**
     * @brief Load IMU data from TUM dataset with time range filtering
     * @param dataset_root_path Root path of the TUM dataset (e.g., /path/to/V1_01_easy)
     * @param start_timestamp_ns Start timestamp in nanoseconds (inclusive)
     * @param end_timestamp_ns End timestamp in nanoseconds (inclusive)
     * @return True if loaded successfully
     */
    static bool load_imu_data_in_range(const std::string& dataset_root_path, 
                                       long long start_timestamp_ns, 
                                       long long end_timestamp_ns);

    /**
     * @brief Get IMU measurements between two timestamps
     * @param start_timestamp Start timestamp in nanoseconds (exclusive)
     * @param end_timestamp End timestamp in nanoseconds (inclusive)
     * @return Vector of IMU measurements between timestamps
     */
    static std::vector<IMUData> get_imu_between_timestamps(long long start_timestamp, long long end_timestamp);

    /**
     * @brief Check if IMU data is loaded
     * @return True if IMU data is available
     */
    static bool has_imu_data();

    /**
     * @brief Print statistics about loaded IMU data
     */
    static void print_imu_stats();

    /**
     * @brief Clear all loaded IMU data
     */
    static void clear_imu_data();

private:
    static std::vector<GroundTruthPose> s_ground_truth_data;
    static bool s_data_loaded;
    
    // Pre-matched poses for image timestamps
    static std::vector<Eigen::Matrix4f> s_matched_poses;
    static std::vector<long long> s_image_timestamps;
    static std::vector<double> s_timestamp_errors;  // In seconds
    
    // IMU data
    static std::vector<IMUData> s_imu_data;
    static bool s_imu_data_loaded;
};

} // namespace lightweight_vio
