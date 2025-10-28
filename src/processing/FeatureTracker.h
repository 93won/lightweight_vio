/**
 * @file      FeatureTracker.h
 * @brief     Defines the feature tracking and management class for VIO.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-11
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include "util/Config.h"

// Forward declarations
namespace lightweight_vio {
    class Frame;
    class Feature;
    class MapPoint;
}

namespace lightweight_vio {

class FeatureTracker {
public:
    FeatureTracker();
    ~FeatureTracker() = default;

    // Main tracking function
    void track_features(std::shared_ptr<Frame> current_frame, 
                       std::shared_ptr<Frame> previous_frame = nullptr);

    // Feature extraction and tracking
    std::pair<int, int> extract_new_features(std::shared_ptr<Frame> frame);
    std::pair<int, int> optical_flow_tracking(std::shared_ptr<Frame> current_frame, 
                              std::shared_ptr<Frame> previous_frame);
    
    // Feature distribution
    void set_mask(std::shared_ptr<Frame> frame);
    
    // Grid-based feature distribution management
    void manage_grid_based_features(std::shared_ptr<Frame> frame);
    void assign_features_to_grid(std::shared_ptr<Frame> frame, std::vector<std::vector<std::vector<int>>>& temp_grid);
    void assign_features_to_grid_with_indices(std::shared_ptr<Frame> frame, 
                                              std::vector<std::vector<std::vector<int>>>& temp_grid,
                                              const std::vector<int>& valid_indices);
    void limit_features_per_grid(std::shared_ptr<Frame> frame, std::vector<std::vector<std::vector<int>>>& temp_grid);
    std::vector<int> select_features_for_tracking(std::shared_ptr<Frame> previous_frame);
    
    // Map point projection
    cv::Point2f project_map_point_to_current_frame(std::shared_ptr<MapPoint> map_point, std::shared_ptr<Frame> current_frame, cv::Point2f prev_pixel = cv::Point2f(-1, -1));
    
    // Velocity-based quality assessment
    void assess_feature_quality_by_velocity(std::shared_ptr<Frame> current_frame);
    
    // Fundamental matrix RANSAC filtering
    void apply_fundamental_matrix_filter(std::shared_ptr<Frame> current_frame, 
                                       std::shared_ptr<Frame> previous_frame);
    
    // ⭐ RGBD-specific 3D computation
    int compute_rgbd_3d(std::shared_ptr<Frame> frame);

private:
    // Configuration reference
    const Config& m_config = Config::getInstance();
    
    // Feature distribution mask
    cv::Mat m_mask;
    
    // Keyframe management state
    double m_last_keyframe_grid_coverage = 0.0;  // Grid coverage of the last keyframe
    
    // Helper functions
    bool is_in_border(const cv::Point2f& point, const cv::Size& img_size, int border_size = 1) const;
    void update_feature_track_count(std::shared_ptr<Frame> frame);
    void update_features_with_points(std::vector<std::shared_ptr<Feature>>& features, 
                                    const std::vector<cv::Point2f>& points,
                                    const std::vector<uchar>& status);
};

} // namespace lightweight_vio
