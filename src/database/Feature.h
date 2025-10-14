/**
 * @file      Feature.h
 * @brief     Defines the Feature class, representing a 2D feature in an image.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-11
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

namespace lightweight_vio {

class Feature {
public:
    Feature(int feature_id, const cv::Point2f& pixel_coord);
    ~Feature() = default;

        // Getters
    int get_feature_id() const { return m_feature_id; }
    cv::Point2f get_pixel_coord() const { return m_pixel_coord; }
    float get_u() const { return m_pixel_coord.x; }
    float get_v() const { return m_pixel_coord.y; }
    cv::Point2f get_undistorted_coord() const { return m_undistorted_coord; }
    Eigen::Vector2f get_normalized_coord() const { return m_normalized_coord; }
    Eigen::Vector3f get_3d_point() const { return m_3d_point; }
    Eigen::Vector2f get_velocity() const { return m_velocity; }
    float get_depth() const { return m_depth; }
    float get_reprojection_error() const { return m_reprojection_error; }
    int get_track_count() const { return m_track_count; }
    int get_num_observations_accumulated() const { return m_num_observations_accumulated; }
    bool is_valid() const { return m_is_valid; }
    bool has_3d_point() const { return m_has_3d_point; }

    // Setters
    void set_pixel_coord(const cv::Point2f& coord) { m_pixel_coord = coord; }
    void set_undistorted_coord(const cv::Point2f& coord) { m_undistorted_coord = coord; }
    void set_normalized_coord(const Eigen::Vector2f& coord) { m_normalized_coord = coord; }
    void set_3d_point(const Eigen::Vector3f& point) { 
        m_3d_point = point; 
        m_has_3d_point = true;
        m_depth = point.z();
    }
    void set_velocity(const Eigen::Vector2f& velocity) { m_velocity = velocity; }
    void set_depth(float depth) { 
        m_depth = depth; 
        
    }
    void set_reprojection_error(float reprojection_error) { m_reprojection_error = reprojection_error; }
    void set_track_count(int count) { m_track_count = count; }
    void set_num_observations_accumulated(int count) { m_num_observations_accumulated = count; }
    void set_valid(bool valid) { m_is_valid = valid; }
    
    // Tracking relationship
    void set_tracked_feature_id(int tracked_id) { m_tracked_feature_id = tracked_id; }
    int get_tracked_feature_id() const { return m_tracked_feature_id; }
    bool has_tracked_feature() const { return m_tracked_feature_id != -1; }

    // Operations
    void increment_track_count() { m_track_count++; }
    void increment_observations_accumulated() { m_num_observations_accumulated++; }
    
    // Stereo operations
    void set_stereo_match(const cv::Point2f& right_coord, float disparity) {
        m_right_coord = right_coord;
        m_disparity = disparity;
        m_has_stereo_match = true;
    }
    
    void set_undistorted_stereo_match(const cv::Point2f& right_undistorted_coord, const Eigen::Vector2f& right_normalized, float undistorted_disparity) {
        m_right_undistorted_coord = right_undistorted_coord;
        m_right_normalized_coord = right_normalized;
        m_undistorted_disparity = undistorted_disparity;
    }
    
    bool has_stereo_match() const { return m_has_stereo_match; }
    const cv::Point2f& get_right_coord() const { return m_right_coord; }
    const cv::Point2f& get_right_undistorted_coord() const { return m_right_undistorted_coord; }
    float get_stereo_disparity() const { return m_disparity; }
    float get_undistorted_disparity() const { return m_undistorted_disparity; }
    Eigen::Vector2f get_right_normalized_coord() const {
        return m_right_normalized_coord;
    }

private:
    int m_feature_id;              // Unique feature ID
    int m_tracked_feature_id;      // ID of the feature this one tracks from previous frame (-1 if none)
    cv::Point2f m_pixel_coord;      // Pixel coordinates in left image
    cv::Point2f m_undistorted_coord; // Undistorted pixel coordinates
    Eigen::Vector2f m_normalized_coord;  // Normalized camera coordinates
    Eigen::Vector3f m_3d_point;    // 3D point in left camera frame
    Eigen::Vector2f m_velocity;    // Optical flow velocity
    int m_track_count;             // Number of times tracked
    int m_num_observations_accumulated; // Total number of observations accumulated (not reset by window sliding)
    float m_depth;                 // Estimated depth (inverse depth parameterization)
    float m_reprojection_error;   // Reprojection error in pixels
    bool m_is_valid;               // Whether this feature is valid
    bool m_has_3d_point;           // Whether 3D point is available
    
    // Stereo matching data
    cv::Point2f m_right_coord;     // Pixel coordinates in right image
    cv::Point2f m_right_undistorted_coord; // Undistorted pixel coordinates in right image
    Eigen::Vector2f m_right_normalized_coord; // Normalized coordinates in right camera
    float m_disparity;             // Stereo disparity (pixel coordinates)
    float m_undistorted_disparity; // Undistorted stereo disparity
    bool m_has_stereo_match;
};

} // namespace lightweight_vio
