/**
 * @file      FeatureTracker.cpp
 * @brief     Implements feature tracking and management for VIO.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-11
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "processing/FeatureTracker.h"
#include "database/Frame.h"
#include "database/Feature.h"
#include "database/MapPoint.h"
#include "util/Config.h"
#include <spdlog/spdlog.h>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <chrono>

namespace lightweight_vio {

FeatureTracker::FeatureTracker() : m_config(Config::getInstance())
{
}

void FeatureTracker::track_features(std::shared_ptr<Frame> current_frame, 
                                   std::shared_ptr<Frame> previous_frame) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    if (!current_frame) {
        std::cerr << "Current frame is null" << std::endl;
        return;
    }

    

    int tracked_features = 0;
    int new_map_points_from_tracking = 0;
    int new_extracted_features = 0;
    int new_map_points_from_extraction = 0;

    auto tracking_time = 0.0;
    auto feature_extraction_time = 0.0;
    auto mask_creation_time = 0.0;

    if (previous_frame) {
        // Track existing features
        auto tracking_start = std::chrono::high_resolution_clock::now();
        auto tracking_stats = optical_flow_tracking(current_frame, previous_frame);
        auto tracking_end = std::chrono::high_resolution_clock::now();
        tracking_time = std::chrono::duration_cast<std::chrono::microseconds>(tracking_end - tracking_start).count() / 1000.0;
        
        tracked_features = tracking_stats.first;
        new_map_points_from_tracking = tracking_stats.second;
        
        // Track count will only be updated when frame becomes keyframe
        // Removed: update_feature_track_count(current_frame);
    }

    // Extract new features if needed
    if (current_frame->get_feature_count() < m_config.m_max_features) {
        auto mask_start = std::chrono::high_resolution_clock::now();
        set_mask(current_frame);
        auto mask_end = std::chrono::high_resolution_clock::now();
        mask_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(mask_end - mask_start).count() / 1000.0;
        
        auto extraction_start = std::chrono::high_resolution_clock::now();
        auto extraction_stats = extract_new_features(current_frame);
        auto extraction_end = std::chrono::high_resolution_clock::now();
        feature_extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(extraction_end - extraction_start).count() / 1000.0;
        
        new_extracted_features = extraction_stats.first;
        new_map_points_from_extraction = extraction_stats.second;
    }

    // Note: Stereo matching is now handled by Frame::compute_stereo_depth() in Estimator
    // This avoids duplicate stereo matching operations
    
    int total_new_map_points = new_map_points_from_tracking + new_map_points_from_extraction;

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    // Count final valid map points
    int final_valid_map_points = 0;
    
    for (size_t i = 0; i < current_frame->get_map_points().size(); ++i) {
        auto mp = current_frame->get_map_point(i);
        if (mp && !mp->is_bad()) {
            final_valid_map_points++;
        }
    }
    
    // After tracking and stereo matching, check if we need more features
    int current_features = current_frame->get_feature_count();
    if (current_features < m_config.m_max_features) {
        // Update mask to avoid detecting features near existing ones
        set_mask(current_frame);
        
        // Extract new features to reach max_features
        auto [new_features, new_successful_matches] = extract_new_features(current_frame);
        new_extracted_features += new_features;
        new_map_points_from_extraction += new_successful_matches;
    }
    
    // Single comprehensive log with timing breakdown
    auto total_time = total_duration.count() / 1000.0;

  
   
    
    // Timing output removed for cleaner logs
}

std::pair<int, int> FeatureTracker::extract_new_features(std::shared_ptr<Frame> frame) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (frame->get_image().empty()) {
        std::cerr << "Cannot extract features: image is empty" << std::endl;
        return {0, 0};
    }

    std::vector<cv::Point2f> corners;
    
    // Use the mask created by set_mask function
    cv::Mat mask_to_use = m_mask.empty() ? cv::Mat() : m_mask.clone();
    
    // For RGBD frames, add depth validity mask
    if (frame->get_frame_type() == FrameType::RGBD) {
        const cv::Mat& depth_map = frame->get_depth_map();
        if (!depth_map.empty()) {
            // Create depth validity mask (255 where depth is valid, 0 where invalid)
            cv::Mat depth_mask = cv::Mat::zeros(depth_map.size(), CV_8UC1);
            
            const auto& config = Config::getInstance();
            for (int y = 0; y < depth_map.rows; ++y) {
                for (int x = 0; x < depth_map.cols; ++x) {
                    float depth = depth_map.at<float>(y, x);
                    // Check if depth is valid (non-zero and within range)
                    if (depth > config.m_min_depth && depth < config.m_max_depth) {
                        depth_mask.at<uchar>(y, x) = 255;
                    }
                }
            }
            
            // Combine with existing mask using AND operation
            if (mask_to_use.empty()) {
                mask_to_use = depth_mask;
            } else {
                cv::bitwise_and(mask_to_use, depth_mask, mask_to_use);
            }
        }
    }

    int features_needed = Config::getInstance().m_max_features - frame->get_feature_count();
    int new_map_points_created = 0;
    
    if (features_needed > 0) {
        auto detection_start = std::chrono::high_resolution_clock::now();
        cv::goodFeaturesToTrack(frame->get_image(), corners, 
                               features_needed,
                               Config::getInstance().m_quality_level,
                               Config::getInstance().m_min_distance,
                               mask_to_use);
        auto detection_end = std::chrono::high_resolution_clock::now();
        auto detection_time = std::chrono::duration_cast<std::chrono::microseconds>(detection_end - detection_start).count() / 1000.0;

        auto feature_processing_start = std::chrono::high_resolution_clock::now();
        
        // Create all features without map points - Estimator will handle map point creation during keyframe creation
        int next_feature_id = frame->get_feature_count();  // Start from current count
        for (const auto& corner : corners) {
            auto feature = std::make_shared<Feature>(next_feature_id++, corner);
            frame->add_feature(feature);
        }
        
        // Return feature count and 0 for map points (no immediate map point creation)
        return std::make_pair(corners.size(), 0);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    return {static_cast<int>(corners.size()), 0};
}

std::pair<int, int> FeatureTracker::optical_flow_tracking(std::shared_ptr<Frame> current_frame,
                                          std::shared_ptr<Frame> previous_frame) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (previous_frame->get_feature_count() == 0) {
        return {0, 0};
    }

    // Extract points from previous frame features using grid-based selection
    auto point_extraction_start = std::chrono::high_resolution_clock::now();
    
    // Get selected feature indices from grid-based selection
    std::vector<int> feature_indices = select_features_for_tracking(previous_frame);
    
    std::vector<cv::Point2f> prev_pts;
    std::vector<int> valid_feature_indices;  // Track valid feature indices for tracking
    
    // Variables for tracking pixel movement from map point projections
    float total_projection_movement = 0.0f;
    int projection_count = 0;

    const auto &prev_features = previous_frame->get_features();
    for (int idx : feature_indices)
    {

        if (!prev_features[idx]->is_valid())
            continue;

        prev_pts.push_back(prev_features[idx]->get_pixel_coord());
        valid_feature_indices.push_back(idx); // Store the original index
    }

    auto point_extraction_end = std::chrono::high_resolution_clock::now();
    auto point_extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(point_extraction_end - point_extraction_start).count() / 1000.0;
    
    std::vector<cv::Point2f> cur_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    // Perform optical flow tracking
    auto flow_start = std::chrono::high_resolution_clock::now();
    int window_size = Config::getInstance().m_window_size;
    
    // Use OpenCV implementation (most stable and optimized)
    cv::calcOpticalFlowPyrLK(previous_frame->get_image(), current_frame->get_image(),
                            prev_pts, cur_pts, status, err,
                            cv::Size(window_size, window_size), 
                            Config::getInstance().m_max_level,
                            Config::getInstance().term_criteria(),
                            0, 0.001);    auto flow_end = std::chrono::high_resolution_clock::now();
    auto flow_time = std::chrono::duration_cast<std::chrono::microseconds>(flow_end - flow_start).count() / 1000.0;

    // Create features for current frame based on tracking results
    auto feature_creation_start = std::chrono::high_resolution_clock::now();
    int tracked_features = 0;
    int associated_map_points = 0;
    int new_map_points_created = 0;
    int next_feature_id = current_frame->get_feature_count();  // Start from current count
    
    // Tracking failure counters for debugging
    int optical_flow_failed = 0;
    int border_failed = 0;
    int error_threshold_failed = 0;
    int movement_exceeded = 0;
    int total_attempted = prev_pts.size();


    
    const Config& config = Config::getInstance();
    int border_size = config.m_border_size;

    
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (!status[i]) {
            optical_flow_failed++;
            continue;
        }
        
        if (!is_in_border(cur_pts[i], current_frame->get_image().size(), border_size)) {
            border_failed++;
            continue;
        }
        
        // Apply optical flow quality checks using config parameters
        float dx = cur_pts[i].x - prev_pts[i].x;
        float dy = cur_pts[i].y - prev_pts[i].y;
        float movement = std::sqrt(dx*dx + dy*dy);
        
        // Check error threshold
        if (err[i] >= Config::getInstance().m_error_threshold) {
            error_threshold_failed++;
            continue;
        }
        
        // Check maximum movement threshold
        if (movement >= Config::getInstance().m_max_movement) {
            movement_exceeded++;
            continue;
        }
        
        // Get the original feature index
        int original_feature_idx = valid_feature_indices[i];
        auto prev_feature = previous_frame->get_features()[original_feature_idx];
        
        // Create new feature with sequential frame-local ID
        auto new_feature = std::make_shared<Feature>(
            next_feature_id++,  // Use sequential ID
            cur_pts[i]
        );
        
        // Set the tracked feature ID to maintain tracking relationship
        new_feature->set_tracked_feature_id(prev_feature->get_feature_id());
        
        // Inherit accumulated observations from previous feature (without incrementing)
        int prev_accumulated_obs = prev_feature->get_num_observations_accumulated();
        new_feature->set_num_observations_accumulated(prev_accumulated_obs);
        
        // Update velocity
        Eigen::Vector2f velocity(dx, dy);
        new_feature->set_velocity(velocity);
        
        // Propagate outlier flag from previous frame
        bool was_outlier = previous_frame->get_outlier_flag(original_feature_idx);
        current_frame->add_feature(new_feature);
        current_frame->set_outlier_flag(current_frame->get_feature_count() - 1, was_outlier);
        new_feature->set_track_count(prev_feature->get_track_count() + 1);
        tracked_features++;
        
        // Check if previous feature has associated map point
        // Use the original feature index to get the correct map point
        auto prev_map_point = previous_frame->get_map_point(original_feature_idx);
        
        if (prev_map_point && !prev_map_point->is_bad()) {
            // Debug: Check if this association looks suspicious
            cv::Point2f current_pt = cur_pts[i];  // Use cur_pts[i] instead of tracked_features[j]
            cv::Point2f prev_pt = prev_pts[i];    // Use prev_pts[i] instead of previous_features
            float tracking_distance = cv::norm(current_pt - prev_pt);
            
            // Associate with existing map point (for pose estimation)
            current_frame->set_map_point(current_frame->get_feature_count() - 1, prev_map_point);
            // NOTE: Do NOT add observation here - observations should only be added when frame becomes keyframe
            // prev_map_point->add_observation(current_frame, current_frame->get_feature_count() - 1);  // REMOVED
            associated_map_points++;
            
            // Map point creation is now handled only by Estimator during keyframe creation
        } 
    }
    auto feature_creation_end = std::chrono::high_resolution_clock::now();
    auto feature_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(feature_creation_end - feature_creation_start).count() / 1000.0;

    // Apply fundamental matrix RANSAC filtering if we have enough features
    if (tracked_features >= 8) {
        apply_fundamental_matrix_filter(current_frame, previous_frame);
    }

    // Perform velocity-based quality assessment
    if (tracked_features > 0) {
        assess_feature_quality_by_velocity(current_frame);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
    
    // Simple projection movement summary
    if (projection_count > 0) {
        float avg_movement = total_projection_movement / projection_count;
        spdlog::info("[FRAME_CHANGE] {} map points projected, avg movement: {:.2f}px", 
                    projection_count, avg_movement);
    }
    
    // Debug output for tracking failures (only when explicitly enabled)
    if (m_config.m_enable_debug_output && false) {  // Disabled tracking debug output
        int total_failed = optical_flow_failed + border_failed + error_threshold_failed + movement_exceeded;
        float success_rate = total_attempted > 0 ? (float)tracked_features / total_attempted * 100.0f : 0.0f;
        
        spdlog::info("[TRACKING DEBUG] Frame-to-frame tracking results:");
        spdlog::info("  Attempted: {} features", total_attempted);
        spdlog::info("  Successful: {} features ({:.1f}%)", tracked_features, success_rate);
        spdlog::info("  Failed breakdown:");
        spdlog::info("    - Optical flow failed: {} ({:.1f}%)", optical_flow_failed, 
                     total_attempted > 0 ? (float)optical_flow_failed / total_attempted * 100.0f : 0.0f);
        spdlog::info("    - Out of border: {} ({:.1f}%)", border_failed,
                     total_attempted > 0 ? (float)border_failed / total_attempted * 100.0f : 0.0f);
        spdlog::info("    - Error threshold (>{:.1f}): {} ({:.1f}%)", 
                     Config::getInstance().m_error_threshold, error_threshold_failed,
                     total_attempted > 0 ? (float)error_threshold_failed / total_attempted * 100.0f : 0.0f);
        spdlog::info("    - Movement exceeded (>{:.1f}px): {} ({:.1f}%)", 
                     Config::getInstance().m_max_movement, movement_exceeded,
                     total_attempted > 0 ? (float)movement_exceeded / total_attempted * 100.0f : 0.0f);
        spdlog::info("  Map point associations: {}", associated_map_points);
    }
    
    return {tracked_features, new_map_points_created};
}

void FeatureTracker::set_mask(std::shared_ptr<Frame> frame) {
    if (frame->get_image().empty()) {
        std::cerr << "Cannot set mask: image is empty" << std::endl;
        return;
    }
    
    // Create mask initialized to 255 (valid areas for feature detection)
    m_mask = cv::Mat(frame->get_image().size(), CV_8UC1, cv::Scalar(255));
    
    const Config& config = Config::getInstance();
    int min_distance = static_cast<int>(config.m_min_distance);
    int border_size = config.m_border_size;
    
    // Set border regions to 0 (invalid for feature detection)
    if (border_size > 0) {
        m_mask(cv::Rect(0, 0, frame->get_image().cols, border_size)) = 0;                    // Top
        m_mask(cv::Rect(0, frame->get_image().rows - border_size, frame->get_image().cols, border_size)) = 0; // Bottom
        m_mask(cv::Rect(0, 0, border_size, frame->get_image().rows)) = 0;                    // Left
        m_mask(cv::Rect(frame->get_image().cols - border_size, 0, border_size, frame->get_image().rows)) = 0; // Right
    }
    
    // For each existing feature, create a circular mask around it (but skip outliers)
    const auto& features = frame->get_features();
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feature = features[i];
        
        // Skip invalid features
        if (!feature->is_valid()) {
            continue;
        }
        
        // Skip outlier features - allow new features to be detected in their place
        if (frame->get_outlier_flag(i)) {
            continue;
        }
        
        cv::Point2f pt = feature->get_pixel_coord();
        
        // Check if feature is within image bounds
        if (pt.x >= 0 && pt.y >= 0 && 
            pt.x < frame->get_image().cols && pt.y < frame->get_image().rows) {
            
            // Create circular mask around existing NON-OUTLIER feature
            cv::circle(m_mask, pt, min_distance, cv::Scalar(0), -1);
        }
    }
    
    // Debug output removed for cleaner logs
}

void FeatureTracker::update_feature_track_count(std::shared_ptr<Frame> frame) {
    for (auto& feature : frame->get_features()) {
        if (feature->is_valid()) {
            feature->set_track_count(feature->get_track_count() + 1);
        }
    }
}

void FeatureTracker::update_features_with_points(
    std::vector<std::shared_ptr<Feature>>& features,
    const std::vector<cv::Point2f>& points,
    const std::vector<uchar>& status) {
    
    size_t point_idx = 0;
    for (auto& feature : features) {
        if (feature->is_valid() && point_idx < points.size()) {
            if (status[point_idx]) {
                feature->set_pixel_coord(points[point_idx]);
            } else {
                feature->set_valid(false);
            }
            point_idx++;
        }
    }
}

bool FeatureTracker::is_in_border(const cv::Point2f& point, const cv::Size& img_size, int border_size) const {
    int img_x = cvRound(point.x);
    int img_y = cvRound(point.y);

    return border_size <= img_x && img_x < img_size.width - border_size && 
           border_size <= img_y && img_y < img_size.height - border_size;
}

void FeatureTracker::assign_features_to_grid(std::shared_ptr<Frame> frame, 
                                             std::vector<std::vector<std::vector<int>>>& temp_grid) {
    const int grid_cols = m_config.m_grid_cols;
    const int grid_rows = m_config.m_grid_rows;
    const int img_width = m_config.m_image_width;
    const int img_height = m_config.m_image_height;
    
    // Assign each feature to its corresponding grid cell
    const auto& features = frame->get_features();
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feature = features[i];
        if (!feature) continue;
        
        // Calculate grid coordinates
        float cell_width = (float)img_width / grid_cols;
        float cell_height = (float)img_height / grid_rows;
        
        cv::Point2f pixel_coord = feature->get_pixel_coord();
        
        // Safety check for pixel coordinates
        if (pixel_coord.x < 0 || pixel_coord.x >= img_width || 
            pixel_coord.y < 0 || pixel_coord.y >= img_height) {
            continue; // Skip features outside image bounds
        }
        
        int grid_x = std::min((int)(pixel_coord.x / cell_width), grid_cols - 1);
        int grid_y = std::min((int)(pixel_coord.y / cell_height), grid_rows - 1);
        
        // Additional safety check for grid indices
        if (grid_x < 0 || grid_x >= grid_cols || grid_y < 0 || grid_y >= grid_rows) {
            continue; // Skip invalid grid coordinates
        }
        
        // Add feature index to the corresponding grid cell
        temp_grid[grid_y][grid_x].push_back(i);
    }
}

void FeatureTracker::assign_features_to_grid_with_indices(std::shared_ptr<Frame> frame, 
                                                          std::vector<std::vector<std::vector<int>>>& temp_grid,
                                                          const std::vector<int>& valid_indices) {
    const int grid_cols = m_config.m_grid_cols;
    const int grid_rows = m_config.m_grid_rows;
    const int img_width = m_config.m_image_width;
    const int img_height = m_config.m_image_height;
    
    // Assign only specified features to their corresponding grid cells
    const auto& features = frame->get_features();
    for (int feature_idx : valid_indices) {
        if (feature_idx < 0 || feature_idx >= features.size()) {
            continue; // Skip invalid indices
        }
        
        const auto& feature = features[feature_idx];
        if (!feature) continue;
        
        // Calculate grid coordinates
        float cell_width = (float)img_width / grid_cols;
        float cell_height = (float)img_height / grid_rows;
        
        cv::Point2f pixel_coord = feature->get_pixel_coord();
        
        // Safety check for pixel coordinates
        if (pixel_coord.x < 0 || pixel_coord.x >= img_width || 
            pixel_coord.y < 0 || pixel_coord.y >= img_height) {
            continue; // Skip features outside image bounds
        }
        
        int grid_x = std::min((int)(pixel_coord.x / cell_width), grid_cols - 1);
        int grid_y = std::min((int)(pixel_coord.y / cell_height), grid_rows - 1);
        
        // Additional safety check for grid indices
        if (grid_x < 0 || grid_x >= grid_cols || grid_y < 0 || grid_y >= grid_rows) {
            continue; // Skip invalid grid coordinates
        }
        
        // Add feature index to the corresponding grid cell
        temp_grid[grid_y][grid_x].push_back(feature_idx);
    }
}

void FeatureTracker::limit_features_per_grid(std::shared_ptr<Frame> frame, 
                                             std::vector<std::vector<std::vector<int>>>& temp_grid) {
    const int max_features_per_grid = m_config.m_max_features_per_grid;
    auto& features = frame->get_features_mutable();
    auto& map_points = frame->get_map_points_mutable();  // Use mutable version
    
    // Process each grid cell
    for (size_t row = 0; row < temp_grid.size(); ++row) {
        for (size_t col = 0; col < temp_grid[row].size(); ++col) {
            auto& cell_features = temp_grid[row][col];

            // Skip if this cell has fewer features than the limit
            if (cell_features.size() <= max_features_per_grid) {
                continue;
            }
            
            // Sort features by track count (as a proxy for strength) in descending order
            // Note: Using track_count since corner_response is not available
            std::sort(cell_features.begin(), cell_features.end(), 
                     [&features](int a, int b) {
                         // Safety checks for valid indices and non-null features
                         if (a >= features.size() || b >= features.size()) return false;
                         if (!features[a] || !features[b]) return false;
                         return features[a]->get_track_count() > features[b]->get_track_count();
                     });
            
            // Keep only the strongest max_features_per_grid features
            // Mark the rest as invalid instead of setting to nullptr
            for (size_t i = max_features_per_grid; i < cell_features.size(); ++i) {
                int feature_idx = cell_features[i];
                
                // Safety check for valid feature index
                if (feature_idx < 0 || feature_idx >= features.size()) {
                    continue; // Skip invalid indices
                }
                
                // Mark feature as invalid instead of setting to nullptr
                if (features[feature_idx]) {
                    features[feature_idx]->set_valid(false);
                }
                
                // Disconnect from map point by removing observation and setting to nullptr
                if (feature_idx < map_points.size()) {
                    auto mp = map_points[feature_idx];
                    if (mp) {
                        // Remove this frame's observation from the map point
                        mp->remove_observation(frame);
                        // Set map point to nullptr in frame
                        map_points[feature_idx] = nullptr;
                    }
                }
            }
            
            // Keep only the selected features in the cell
            cell_features.resize(max_features_per_grid);
        }
    }
    
    if (m_config.m_enable_debug_output) {
        // Count remaining valid and connected features
        auto& features = frame->get_features_mutable();
        auto& map_points = frame->get_map_points_mutable();
        
        int valid_features = 0;
        int connected_features = 0;
        
        size_t max_size = std::min(features.size(), map_points.size());
        
        for (size_t i = 0; i < features.size(); ++i) {
            if (features[i] && features[i]->is_valid()) {
                valid_features++;
                if (i < map_points.size() && map_points[i] && !map_points[i]->is_bad()) {
                    connected_features++;
                }
            }
        }
        
        spdlog::info("Grid-based feature management: {} valid features, {} connected to map points", 
                     valid_features, connected_features);
    }
}

std::vector<int> FeatureTracker::select_features_for_tracking(std::shared_ptr<Frame> previous_frame) {
    std::vector<int> selected_indices;
    
    if (!previous_frame || previous_frame->get_feature_count() == 0) {
        return selected_indices;
    }
    
    // Simply collect all valid features without any observation count filtering
    const auto& features = previous_frame->get_features();
    std::vector<int> valid_feature_indices;
    
    for (int i = 0; i < features.size(); ++i) {
        if (!features[i] || !features[i]->is_valid()) {
            continue;
        }
        
        valid_feature_indices.push_back(i);
    }
    
    // Initialize grid
    const int grid_rows = m_config.m_grid_rows;
    const int grid_cols = m_config.m_grid_cols;
    const int max_features_per_grid = m_config.m_max_features_per_grid;
    
    std::vector<std::vector<std::vector<int>>> temp_grid(grid_rows, 
                                                        std::vector<std::vector<int>>(grid_cols));
    
    // Assign only valid features to grid cells
    assign_features_to_grid_with_indices(previous_frame, temp_grid, valid_feature_indices);
    
    // Select best features from each grid cell
    for (size_t row = 0; row < temp_grid.size(); ++row) {
        for (size_t col = 0; col < temp_grid[row].size(); ++col) {
            auto& cell_features = temp_grid[row][col];
            
            if (cell_features.empty()) {
                continue;
            }
            
            const auto& features = previous_frame->get_features();
            
            // Sort features by track count (as a proxy for strength) in descending order
            std::sort(cell_features.begin(), cell_features.end(), 
                     [&features](int a, int b) {
                         // Safety checks for valid indices and non-null features
                         if (a >= features.size() || b >= features.size()) return false;
                         if (!features[a] || !features[b]) return false;
                         return features[a]->get_track_count() > features[b]->get_track_count();
                     });
            
            // Select up to max_features_per_grid best features from this cell
            int features_to_select = std::min((int)cell_features.size(), max_features_per_grid);
            for (int i = 0; i < features_to_select; ++i) {
                int feature_idx = cell_features[i];
                if (feature_idx >= 0 && feature_idx < features.size() && features[feature_idx]) {
                    selected_indices.push_back(feature_idx);
                }
            }
        }
    }
    
    return selected_indices;
}

cv::Point2f FeatureTracker::project_map_point_to_current_frame(std::shared_ptr<MapPoint> map_point, std::shared_ptr<Frame> current_frame, cv::Point2f prev_pixel) {
    if (!map_point || !current_frame) {
        return cv::Point2f(-1, -1);  // Invalid projection
    }
    
    // Get 3D world point from map point
    Eigen::Vector3f world_point = map_point->get_position();
    
    // Get current frame pose (T_wb: body to world)
    Eigen::Matrix4f T_wb = current_frame->get_Twb();
    
    // Get T_CB (body to camera) from frame
    Eigen::Matrix4d T_CB_double = current_frame->get_Tcb();
    Eigen::Matrix4f T_CB = T_CB_double.cast<float>();
    
    // Calculate T_cw = T_cb * T_bw = T_cb * T_wb.inverse()
    Eigen::Matrix4f T_cw = T_CB * T_wb.inverse();
    
    // Transform world point to camera coordinates
    Eigen::Vector4f world_point_homo(world_point.x(), world_point.y(), world_point.z(), 1.0f);
    Eigen::Vector4f camera_point_homo = T_cw * world_point_homo;
    
    // Check if point is in front of camera
    if (camera_point_homo.z() <= 0) {
        return cv::Point2f(-1, -1);  // Point behind camera
    }
    
    // Convert to 3D camera coordinates
    Eigen::Vector3f camera_point = camera_point_homo.head<3>() / camera_point_homo.w();
    
    // Get camera intrinsics from frame
    float fx = current_frame->get_fx();
    float fy = current_frame->get_fy();
    float cx = current_frame->get_cx();
    float cy = current_frame->get_cy();
    
    // Project to pixel coordinates (assuming pinhole model)
    cv::Point2f pixel_undistorted;
    pixel_undistorted.x = (camera_point.x() / camera_point.z()) * fx + cx;
    pixel_undistorted.y = (camera_point.y() / camera_point.z()) * fy + cy;
    
    // Calculate pixel movement
    float pixel_movement = cv::norm(pixel_undistorted - prev_pixel);
    

    // image boundary check from calculated corner

    
    return pixel_undistorted;
}

void FeatureTracker::assess_feature_quality_by_velocity(std::shared_ptr<Frame> current_frame) {
    if (!current_frame || current_frame->get_feature_count() == 0) {
        return;
    }
    
    std::vector<float> velocity_magnitudes;
    
    // Collect all velocity magnitudes from tracked features
    const auto& features = current_frame->get_features();
    for (const auto& feature : features) {
        if (feature && feature->is_valid() && feature->get_track_count() > 1) {
            float vel_mag = feature->get_velocity().norm();
            velocity_magnitudes.push_back(vel_mag);
        }
    }
    
    if (velocity_magnitudes.empty()) {
        return; // No tracked features to analyze
    }
    
    // Calculate velocity statistics
    std::sort(velocity_magnitudes.begin(), velocity_magnitudes.end());
    float median_velocity = velocity_magnitudes[velocity_magnitudes.size() / 2];
    float velocity_threshold = median_velocity * 3.0f; // 3-sigma rule
    
    // Count outliers and log statistics
    int outlier_count = 0;
    int total_tracked = 0;
    
    // Filter out features with abnormal velocity
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feature = features[i];
        if (feature && feature->is_valid() && feature->get_track_count() > 1) {
            total_tracked++;
            float vel_mag = feature->get_velocity().norm();
            if (vel_mag > velocity_threshold) {
                // Mark as velocity outlier in frame
                current_frame->set_outlier_flag(i, true);
                outlier_count++;
                
                // if (m_config.m_enable_debug_output && outlier_count <= 5) {
                //     spdlog::debug("Feature {} marked as velocity outlier: {:.2f}px (threshold: {:.2f}px)", 
                //                  feature->get_feature_id(), vel_mag, velocity_threshold);
                // }
            }
        }
    }
    
    // if (m_config.m_enable_debug_output && outlier_count > 0) {
    //     spdlog::info("Velocity outlier detection: {}/{} features marked as outliers (median_vel: {:.2f}px, threshold: {:.2f}px)", 
    //                  outlier_count, total_tracked, median_velocity, velocity_threshold);
    // }
}

void FeatureTracker::apply_fundamental_matrix_filter(std::shared_ptr<Frame> current_frame, 
                                                    std::shared_ptr<Frame> previous_frame) {
    // Collect matched feature points
    std::vector<cv::Point2f> cur_pts, prev_pts;
    std::vector<int> feature_indices;
    
    const auto& cur_features = current_frame->get_features();
    const auto& prev_features = previous_frame->get_features();
    
    // Find matched features by tracked_feature_id
    for (int i = 0; i < cur_features.size(); ++i) {
        auto cur_feature = cur_features[i];
        if (!cur_feature || !cur_feature->is_valid()) continue;
        
        int tracked_id = cur_feature->get_tracked_feature_id();
        if (tracked_id < 0) continue;
        
        // Find corresponding feature in previous frame
        for (int j = 0; j < prev_features.size(); ++j) {
            auto prev_feature = prev_features[j];
            if (prev_feature && prev_feature->is_valid() && 
                prev_feature->get_feature_id() == tracked_id) {
                
                cur_pts.push_back(cur_feature->get_pixel_coord());
                prev_pts.push_back(prev_feature->get_pixel_coord());
                feature_indices.push_back(i);
                break;
            }
        }
    }
    
    // Need at least 8 points for fundamental matrix estimation
    if (cur_pts.size() < 8) {
        return;
    }
    
    // Apply fundamental matrix RANSAC filtering 
    std::vector<uchar> status;
    cv::Mat fundamental_matrix = cv::findFundamentalMat(
        prev_pts, cur_pts, 
        cv::FM_RANSAC, 
        m_config.m_F_threshold,  // Distance threshold (pixels)
        0.99,                    // Confidence level
        status
    );
    
    // Mark outliers based on fundamental matrix RANSAC
    int outlier_count = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] == 0) {  // RANSAC outlier
            current_frame->set_outlier_flag(feature_indices[i], true);
            outlier_count++;
        }
    }
    
    // if (m_config.m_enable_debug_output && outlier_count > 0) {
    //     spdlog::info("Fundamental matrix RANSAC: {}/{} features marked as outliers (threshold: {:.1f}px)", 
    //                  outlier_count, status.size(), m_config.m_F_threshold);
    // }
}

// ⭐ ========================================================================
// RGBD-specific 3D computation
// ========================================================================

int FeatureTracker::compute_rgbd_3d(std::shared_ptr<Frame> frame) {
    if (!frame) {
        spdlog::warn("Frame is null in compute_rgbd_3d");
        return 0;
    }
    
    if (!frame->is_rgbd()) {
        spdlog::warn("Frame {} is not RGBD type", frame->get_frame_id());
        return 0;
    }
    
    // Get camera intrinsics
    double fx, fy, cx, cy;
    fx = frame->get_fx(); fy = frame->get_fy(); cx = frame->get_cx(); cy = frame->get_cy();
    
    int valid_3d_count = 0;
    auto& features = frame->get_features_mutable();
    
    for (auto& feature : features) {
        if (!feature) continue;
        
        // Get 2D pixel coordinates using getter
        float u = feature->get_u();
        float v = feature->get_v();
        
        // Get depth from depth map (with interpolation)
        float depth = frame->get_interpolated_depth(u, v);
        
        // Check if depth is valid
        if (depth <= 0.0f) {
            continue; // Skip invalid depth, feature->has_3d_point() will return false
        }
        
        // ⭐ Back-projection: pixel + depth → 3D point
        // X = (u - cx) * depth / fx
        // Y = (v - cy) * depth / fy
        // Z = depth
        float X = (u - cx) * depth / fx;
        float Y = (v - cy) * depth / fy;
        float Z = depth;
        
        // Set 3D coordinates using setter
        feature->set_3d_point(Eigen::Vector3f(X, Y, Z));
        
        // Compute uncertainty (optional, for later use)
        float uncertainty = frame->compute_depth_uncertainty(depth);
        // Can store uncertainty in feature if needed
        
        valid_3d_count++;
    }
    
    return valid_3d_count;
}

} // namespace lightweight_vio
