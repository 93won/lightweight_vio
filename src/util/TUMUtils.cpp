/**
 * @file      TUMUtils.cpp
 * @brief     Implements utilities for handling the TUM dataset.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-09-27
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "util/TUMUtils.h"
#include "database/Frame.h" // For IMUData struct
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <optional>
#include <spdlog/spdlog.h>

namespace lightweight_vio {

// Static member definitions
std::vector<TUMUtils::GroundTruthPose> TUMUtils::s_ground_truth_data;
bool TUMUtils::s_data_loaded = false;

// Pre-matched pose data
std::vector<Eigen::Matrix4f> TUMUtils::s_matched_poses;
std::vector<long long> TUMUtils::s_image_timestamps;
std::vector<double> TUMUtils::s_timestamp_errors;

// IMU data static members
std::vector<IMUData> TUMUtils::s_imu_data;
bool TUMUtils::s_imu_data_loaded = false;

bool TUMUtils::load_ground_truth(const std::string& dataset_root_path) {
    std::string gt_file_path = dataset_root_path + "/mav0/mocap0/data.csv";
    
    std::ifstream file(gt_file_path);
    if (!file.is_open()) {
        spdlog::error("[TUMUtils] Failed to open ground truth file: {}", gt_file_path);
        return false;
    }
    
    std::string line;
    // Skip header line
    if (!std::getline(file, line)) {
        spdlog::error("[TUMUtils] Ground truth file is empty");
        return false;
    }
    
    s_ground_truth_data.clear();
    
    int line_count = 0;
    while (std::getline(file, line)) {
        line_count++;
        
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        std::string token;
        std::vector<double> values;
        
        // Parse CSV values (comma-separated)
        while (std::getline(iss, token, ',')) {
            try {
                values.push_back(std::stod(token));
            } catch (const std::exception& e) {
                spdlog::error("[TUMUtils] Failed to parse value '{}' in line {}: {}", token, line_count, e.what());
                continue;
            }
        }
        
        if (values.size() < 8) {  // timestamp + 3 position + 4 quaternion (TUM format)
            spdlog::warn("[TUMUtils] Incomplete data in line {}, got {} values, expected 8", line_count, values.size());
            continue;
        }
        
        GroundTruthPose gt_pose;
        
        // Extract data: timestamp, px, py, pz, qw, qx, qy, qz (EuRoC format)
        gt_pose.timestamp = static_cast<long long>(values[0]);
        Eigen::Vector3f position(values[1], values[2], values[3]);
        Eigen::Quaternionf quaternion(values[4], values[5], values[6], values[7]);  // qw,qx,qy,qz format
        
        // Create 4x4 transformation matrix (T_WS - world to sensor)
        Eigen::Matrix4f T_WS = Eigen::Matrix4f::Identity();
        T_WS.block<3,3>(0,0) = quaternion.toRotationMatrix();
        T_WS.block<3,1>(0,3) = position;

        // T_WS = T_WS.inverse();
        
       
        
        gt_pose.pose = T_WS;
        
        // EuRoC dataset doesn't provide velocity and bias data in GT file, set to zero
        gt_pose.velocity = Eigen::Vector3f::Zero();
        gt_pose.bias_gyro = Eigen::Vector3f::Zero();
        gt_pose.bias_accel = Eigen::Vector3f::Zero();
        
        s_ground_truth_data.push_back(gt_pose);
    }
    
    file.close();
    
    // // Transform all T_WS poses relative to the first T_WS pose (make first T_WS Identity)
    // if (!s_ground_truth_data.empty()) {
    //     // First, extract all T_WS poses and find the first one
    //     std::vector<Eigen::Matrix4f> original_T_WS_poses;
    //     for (const auto& gt_pose : s_ground_truth_data) {
    //         // Reverse the T_SB transformation to get back original T_WS
            
    //         Eigen::Matrix4f T_WS = gt_pose.pose;
    //         original_T_WS_poses.push_back(T_WS);
    //     }
        
    //     // Get inverse of first T_WS
    //     Eigen::Matrix4f first_T_WS_inv = original_T_WS_poses[0].inverse();
        
    
        
    //     for (size_t i = 0; i < s_ground_truth_data.size(); ++i) {
    //         // Make T_WS relative to first T_WS
    //         Eigen::Matrix4f relative_T_WS = original_T_WS_poses[i] * first_T_WS_inv;
            
    //         // Apply T_SB transformation
    //         s_ground_truth_data[i].pose = relative_T_WS;
    //     }
        
    //     spdlog::info("[TUMUtils] Transformed all GT poses relative to first T_WS pose (now Identity)");
    //     spdlog::info("[TUMUtils] First pose after transformation:");
    //     auto& first = s_ground_truth_data[0].pose;
    //     spdlog::info("[TUMUtils]   [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", first(0,0), first(0,1), first(0,2), first(0,3));
    //     spdlog::info("[TUMUtils]   [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", first(1,0), first(1,1), first(1,2), first(1,3));
    //     spdlog::info("[TUMUtils]   [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", first(2,0), first(2,1), first(2,2), first(2,3));
    //     spdlog::info("[TUMUtils]   [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", first(3,0), first(3,1), first(3,2), first(3,3));
    // }
    
    s_data_loaded = true;
    
    return true;
}

Eigen::Matrix4f TUMUtils::get_ground_truth_pose(long long timestamp) {
    if (!s_data_loaded || s_ground_truth_data.empty()) {
        spdlog::warn("[TUMUtils] No ground truth data loaded");
        return Eigen::Matrix4f::Identity();
    }
    
    // Find closest timestamp using binary search
    auto compare = [](const GroundTruthPose& pose, long long ts) {
        return pose.timestamp < ts;
    };
    
    auto it = std::lower_bound(s_ground_truth_data.begin(), s_ground_truth_data.end(), 
                              timestamp, compare);
    
    size_t index = 0;
    if (it == s_ground_truth_data.end()) {
        // Use last pose if timestamp is beyond the last ground truth
        index = s_ground_truth_data.size() - 1;
    } else if (it == s_ground_truth_data.begin()) {
        // Use first pose if timestamp is before the first ground truth
        index = 0;
    } else {
        // Choose between current and previous based on which is closer
        size_t curr_idx = std::distance(s_ground_truth_data.begin(), it);
        size_t prev_idx = curr_idx - 1;
        
        long long curr_diff = std::abs(s_ground_truth_data[curr_idx].timestamp - timestamp);
        long long prev_diff = std::abs(s_ground_truth_data[prev_idx].timestamp - timestamp);
        
        index = (curr_diff < prev_diff) ? curr_idx : prev_idx;
    }
    
    long long time_diff = std::abs(s_ground_truth_data[index].timestamp - timestamp);
    double time_diff_sec = time_diff / 1e9;
    double image_time_sec = timestamp / 1e9;
    double gt_time_sec = s_ground_truth_data[index].timestamp / 1e9;
    
    if (time_diff > 50000000) {  // 50ms threshold
        spdlog::warn("[TUMUtils] Large timestamp difference: {:.3f} sec (image: {:.6f}, GT: {:.6f})", 
                    time_diff_sec, image_time_sec, gt_time_sec);
    } else {
        // Print successful match info for debugging
        spdlog::debug("[TUMUtils] Matched timestamp: image {:.6f} -> GT {:.6f} (diff: {:.6f} sec, index: {})", 
                     image_time_sec, gt_time_sec, time_diff_sec, index);
    }
    
    return s_ground_truth_data[index].pose;
}

Eigen::Matrix4f TUMUtils::get_ground_truth_pose_sec(double timestamp_sec) {
    // Convert seconds to nanoseconds and call the main function
    long long timestamp_ns = static_cast<long long>(timestamp_sec * 1e9);
    return get_ground_truth_pose(timestamp_ns);
}

void TUMUtils::print_ground_truth_stats() {
    if (!s_data_loaded || s_ground_truth_data.empty()) {
        return;
    }
    
    // Statistics calculation without logging
}

bool TUMUtils::has_ground_truth() {
    return s_data_loaded && !s_ground_truth_data.empty();
}

bool TUMUtils::match_image_timestamps(const std::vector<long long>& image_timestamps) {
    if (!s_data_loaded || s_ground_truth_data.empty()) {
        spdlog::error("[TUMUtils] No ground truth data loaded for matching");
        return false;
    }
    
    if (image_timestamps.empty()) {
        spdlog::error("[TUMUtils] No image timestamps provided for matching");
        return false;
    }
    
    // Define GT time range with buffer (5ms = 5,000,000 ns)
    long long gt_start_time = s_ground_truth_data.front().timestamp;
    long long gt_end_time = s_ground_truth_data.back().timestamp;
    const long long time_threshold_ns = 5000000; // 5ms in nanoseconds
    
    s_matched_poses.clear();
    s_image_timestamps.clear();
    s_timestamp_errors.clear();
    
    // Reserve space for maximum possible matches
    s_matched_poses.reserve(image_timestamps.size());
    s_image_timestamps.reserve(image_timestamps.size());
    s_timestamp_errors.reserve(image_timestamps.size());
    
    int skipped_before_gt = 0;
    int skipped_after_gt = 0;
    int skipped_large_error = 0;
    int matched_count = 0;
    
    for (size_t i = 0; i < image_timestamps.size(); ++i) {
        long long image_ts = image_timestamps[i];
        
        // Skip images that are outside GT time range
        if (image_ts < gt_start_time - time_threshold_ns) {
            skipped_before_gt++;
            continue;
        }
        if (image_ts > gt_end_time + time_threshold_ns) {
            skipped_after_gt++;
            continue;
        }
        
        // Find closest GT timestamp using binary search
        auto compare = [](const GroundTruthPose& pose, long long ts) {
            return pose.timestamp < ts;
        };
        
        auto it = std::lower_bound(s_ground_truth_data.begin(), s_ground_truth_data.end(), 
                                  image_ts, compare);
        
        size_t index = 0;
        if (it == s_ground_truth_data.end()) {
            index = s_ground_truth_data.size() - 1;
        } else if (it == s_ground_truth_data.begin()) {
            index = 0;
        } else {
            size_t curr_idx = std::distance(s_ground_truth_data.begin(), it);
            size_t prev_idx = curr_idx - 1;
            
            long long curr_diff = std::abs(s_ground_truth_data[curr_idx].timestamp - image_ts);
            long long prev_diff = std::abs(s_ground_truth_data[prev_idx].timestamp - image_ts);
            
            index = (curr_diff < prev_diff) ? curr_idx : prev_idx;
        }
        
        // Check if the match is within acceptable threshold
        long long time_diff_ns = std::abs(s_ground_truth_data[index].timestamp - image_ts);
        if (time_diff_ns > time_threshold_ns) {
            skipped_large_error++;
            continue;
        }
        
        // Store matched data
        s_matched_poses.push_back(s_ground_truth_data[index].pose);
        s_image_timestamps.push_back(image_ts);
        
        double time_error_sec = time_diff_ns / 1e9;
        s_timestamp_errors.push_back(time_error_sec);
        
        matched_count++;
    }
    
    // Print statistics
    if (s_timestamp_errors.empty()) {
        spdlog::warn("[TUMUtils] No valid matches found within time threshold");
        return false;
    }
    
    double max_error = *std::max_element(s_timestamp_errors.begin(), s_timestamp_errors.end());
    double avg_error = 0.0;
    for (double err : s_timestamp_errors) avg_error += err;
    avg_error /= s_timestamp_errors.size();
    
    // Statistics calculation without logging
    
    return true;
}

std::optional<Eigen::Matrix4f> TUMUtils::get_matched_pose(size_t image_index) {
    if (image_index >= s_matched_poses.size()) {
        spdlog::error("[TUMUtils] Invalid image index {} (max: {})", image_index, s_matched_poses.size() - 1);
        return std::nullopt;
    }
    
    return s_matched_poses[image_index];
}

size_t TUMUtils::get_matched_count() {
    return s_matched_poses.size();
}

long long TUMUtils::get_matched_timestamp(size_t index) {
    if (index >= s_image_timestamps.size()) {
        return 0;
    }
    return s_image_timestamps[index];
}

// Helper function to trim whitespace
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

bool TUMUtils::load_imu_data(const std::string& dataset_root_path) {
    std::string imu_file_path = dataset_root_path + "/mav0/imu0/data.csv";
    
    std::ifstream file(imu_file_path);
    if (!file.is_open()) {
        spdlog::error("[TUMUtils] Failed to open IMU file: {}", imu_file_path);
        return false;
    }
    
    std::string line;
    // Skip header line
    if (!std::getline(file, line)) {
        spdlog::error("[TUMUtils] IMU file is empty");
        return false;
    }
    
    s_imu_data.clear();
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;  // Skip empty lines or comments
        }
        
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        
        // Parse CSV line
        while (std::getline(ss, cell, ',')) {
            row.push_back(trim(cell));
        }
        
        // EuRoC IMU format: timestamp,w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z
        // where w_RS_S_* are angular velocities and a_RS_S_* are linear accelerations
        if (row.size() >= 7) {
            try {
                long long timestamp_ns = std::stoll(row[0]);
                double timestamp_sec = static_cast<double>(timestamp_ns) / 1e9;
                
                // Parse angular velocity [rad/s]
                float gyro_x = std::stof(row[1]);
                float gyro_y = std::stof(row[2]);
                float gyro_z = std::stof(row[3]);

                // Parse linear acceleration [m/s^2]
                float accel_x = std::stof(row[4]);
                float accel_y = std::stof(row[5]);
                float accel_z = std::stof(row[6]);
                
                // Create IMU data
                IMUData imu_data;
                imu_data.timestamp = timestamp_sec;
                imu_data.angular_vel = Eigen::Vector3f(gyro_x, gyro_y, gyro_z);
                imu_data.linear_accel = Eigen::Vector3f(accel_x, accel_y, accel_z);
                
                s_imu_data.push_back(imu_data);
                
            } catch (const std::exception& e) {
                spdlog::warn("[TUMUtils] Failed to parse IMU line: {} - {}", line, e.what());
                continue;
            }
        }
    }
    
    file.close();
    
    if (s_imu_data.empty()) {
        spdlog::error("[TUMUtils] No valid IMU data loaded");
        s_imu_data_loaded = false;
        return false;
    }
    
    // Sort by timestamp (should already be sorted, but just in case)
    std::sort(s_imu_data.begin(), s_imu_data.end(), 
              [](const IMUData& a, const IMUData& b) {
                  return a.timestamp < b.timestamp;
              });
    
    s_imu_data_loaded = true;
    
    return true;
}

bool TUMUtils::load_imu_data_in_range(const std::string& dataset_root_path, 
                                       long long start_timestamp_ns, 
                                       long long end_timestamp_ns) {
    // First load all IMU data
    if (!load_imu_data(dataset_root_path)) {
        return false;
    }
    
    // Filter IMU data to the specified time range
    std::vector<IMUData> filtered_imu_data;
    
    double start_sec = static_cast<double>(start_timestamp_ns) / 1e9;
    double end_sec = static_cast<double>(end_timestamp_ns) / 1e9;
    
    for (const auto& imu : s_imu_data) {
        if (imu.timestamp >= start_sec && imu.timestamp <= end_sec) {
            filtered_imu_data.push_back(imu);
        }
    }
    
    // Replace the full IMU data with filtered data
    s_imu_data = std::move(filtered_imu_data);
    
    if (s_imu_data.empty()) {
        spdlog::warn("[TUMUtils] No IMU data found in specified time range [{:.6f}, {:.6f}] sec", start_sec, end_sec);
        s_imu_data_loaded = false;
        return false;
    }
    
    spdlog::info("[TUMUtils] Loaded {} IMU samples in time range [{:.6f}, {:.6f}] sec", 
                 s_imu_data.size(), start_sec, end_sec);
    
    return true;
}

std::vector<IMUData> TUMUtils::get_imu_between_timestamps(long long start_timestamp, long long end_timestamp) {
    std::vector<IMUData> result;
    
    if (!s_imu_data_loaded || s_imu_data.empty()) {
        return result;
    }
    
    // Convert nanoseconds to seconds
    double start_sec = static_cast<double>(start_timestamp) / 1e9;
    double end_sec = static_cast<double>(end_timestamp) / 1e9;
    
    // Find IMU data between timestamps (exclusive start, inclusive end)
    for (const auto& imu : s_imu_data) {
        if (imu.timestamp > start_sec && imu.timestamp <= end_sec) {
            result.push_back(imu);
        }
        // Early exit if we've passed the end time (since data is sorted)
        if (imu.timestamp > end_sec) {
            break;
        }
    }
    
    return result;
}

bool TUMUtils::has_imu_data() {
    return s_imu_data_loaded && !s_imu_data.empty();
}

void TUMUtils::print_imu_stats() {
    if (!s_imu_data_loaded || s_imu_data.empty()) {
        spdlog::info("[TUMUtils] No IMU data loaded");
        return;
    }
    
   
    
    double duration = s_imu_data.back().timestamp - s_imu_data.front().timestamp;
    double avg_frequency = (s_imu_data.size() - 1) / duration;
    
    // Calculate some basic statistics
    Eigen::Vector3f accel_sum = Eigen::Vector3f::Zero();
    Eigen::Vector3f gyro_sum = Eigen::Vector3f::Zero();

    for (const auto& imu : s_imu_data) {
        accel_sum += imu.linear_accel;
        gyro_sum += imu.angular_vel;
    }
    
    Eigen::Vector3f accel_mean = accel_sum / s_imu_data.size();
    Eigen::Vector3f gyro_mean = gyro_sum / s_imu_data.size();
    
    
}

void TUMUtils::clear_imu_data() {
    s_imu_data.clear();
    s_imu_data_loaded = false;
    spdlog::info("[TUMUtils] IMU data cleared");
}

} // namespace lightweight_vio
