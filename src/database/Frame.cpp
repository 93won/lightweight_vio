/**
 * @file      Frame.cpp
 * @brief     Implements the Frame class, representing a single camera capture.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-11
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "database/Frame.h"
#include "database/Feature.h" // Include Feature header
#include "database/MapPoint.h"
#include "util/Config.h"
#include "processing/IMUHandler.h" // Include for IMUPreintegration
#include <opencv2/features2d.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include <numeric>
#include <algorithm>

namespace lightweight_vio {

// Define the destructor here, where Feature is a complete type
Frame::~Frame() = default;

// Static member definition
std::shared_ptr<Frame> Frame::m_last_keyframe = nullptr;

Frame::Frame(long long timestamp, int frame_id)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_rotation(Eigen::Matrix3f::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_is_keyframe(false)
    , m_world_pose(Sophus::SE3f())  // Initialize as identity
    , m_velocity(Eigen::Vector3f::Zero())  // Initialize velocity as zero
    , m_accel_bias(Eigen::Vector3f::Zero())  // Initialize accel bias as zero
    , m_gyro_bias(Eigen::Vector3f::Zero())   // Initialize gyro bias as zero
    , m_dt_from_last_keyframe(0.0)          // Initialize dt as zero
    , m_T_relative_from_ref(Eigen::Matrix4f::Identity())
    , m_fx(500.0), m_fy(500.0)  // Default focal lengths
    , m_cx(320.0), m_cy(240.0)  // Default principal point
{
    // Initialize default distortion coefficients (no distortion)
    m_distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};
    
    // Get T_BC from config and convert to T_CB (body to camera)
    const Config& config = Config::getInstance();
    cv::Mat T_bc_cv = config.left_T_BC();  // T_BC (camera to body)
    Eigen::Matrix4d T_bc;  // T_BC (camera to body)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T_bc(i, j) = T_bc_cv.at<double>(i, j);
        }
    }
    m_T_CB = T_bc.inverse();  // Convert T_BC to T_CB (body to camera)
    
    // Set reference keyframe to last keyframe if available
    if (m_last_keyframe) {
        m_reference_keyframe = m_last_keyframe;
        // Initialize with identity transform (will be updated by tracking)
        m_T_relative_from_ref = Eigen::Matrix4f::Identity();
    }
    
    // Calculate undistorted border limits
    calculate_border();
}

Frame::Frame(long long timestamp, int frame_id, 
             double fx, double fy, double cx, double cy, 
             const std::vector<double>& distortion_coeffs)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_rotation(Eigen::Matrix3f::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_is_keyframe(false)
    , m_world_pose(Sophus::SE3f())  // Initialize as identity
    , m_velocity(Eigen::Vector3f::Zero())  // Initialize velocity as zero
    , m_accel_bias(Eigen::Vector3f::Zero())  // Initialize accel bias as zero
    , m_gyro_bias(Eigen::Vector3f::Zero())   // Initialize gyro bias as zero
    , m_dt_from_last_keyframe(0.0)          // Initialize dt as zero
    , m_T_relative_from_ref(Eigen::Matrix4f::Identity())
    , m_fx(fx), m_fy(fy)
    , m_cx(cx), m_cy(cy)
    , m_distortion_coeffs(distortion_coeffs)
{
    // Get T_BC from config and convert to T_CB (body to camera)
    const Config& config = Config::getInstance();
    cv::Mat T_BC_cv = config.left_T_BC();  // T_BC (camera to body)
    Eigen::Matrix4d T_BC;  // T_BC (camera to body)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T_BC(i, j) = T_BC_cv.at<double>(i, j);
        }
    }
    m_T_CB = T_BC.inverse();  // Convert T_BC to T_CB (body to camera)
    
    // Set reference keyframe to last keyframe if available
    if (m_last_keyframe) {
        m_reference_keyframe = m_last_keyframe;
        m_T_relative_from_ref = Eigen::Matrix4f::Identity();
    }
    
    // Calculate undistorted border limits
    calculate_border();
}

Frame::Frame(long long timestamp, int frame_id,
             const cv::Mat& left_image, const cv::Mat& right_image,
             double fx, double fy, double cx, double cy, 
             const std::vector<double>& distortion_coeffs)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_left_image(left_image.clone())
    , m_right_image(right_image.clone())
    , m_rotation(Eigen::Matrix3f::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_is_keyframe(false)
    , m_world_pose(Sophus::SE3f())  // Initialize as identity
    , m_velocity(Eigen::Vector3f::Zero())  // Initialize velocity as zero
    , m_accel_bias(Eigen::Vector3f::Zero())  // Initialize accel bias as zero
    , m_gyro_bias(Eigen::Vector3f::Zero())   // Initialize gyro bias as zero
    , m_dt_from_last_keyframe(0.0)          // Initialize dt as zero
    , m_T_relative_from_ref(Eigen::Matrix4f::Identity())
    , m_fx(fx), m_fy(fy)
    , m_cx(cx), m_cy(cy)
    , m_distortion_coeffs(distortion_coeffs)
{
    // Get T_BC from config and convert to T_CB (body to camera)
    const Config& config = Config::getInstance();
    cv::Mat T_bc_cv = config.left_T_BC();  // T_BC (camera to body)
    Eigen::Matrix4d T_bc;  // T_BC (camera to body)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T_bc(i, j) = T_bc_cv.at<double>(i, j);
        }
    }
    m_T_CB = T_bc.inverse();  // Convert T_BC to T_CB (body to camera)
    
    // Set reference keyframe to last keyframe if available
    if (m_last_keyframe) {
        m_reference_keyframe = m_last_keyframe;
        m_T_relative_from_ref = Eigen::Matrix4f::Identity();
    }
    
    // Calculate undistorted border limits
    calculate_border();
}

Frame::Frame(long long timestamp, int frame_id,
             const cv::Mat& left_image, const cv::Mat& right_image)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_left_image(left_image.clone())
    , m_right_image(right_image.clone())
    , m_rotation(Eigen::Matrix3f::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_is_keyframe(false)
    , m_T_relative_from_ref(Eigen::Matrix4f::Identity())
    , m_accel_bias(Eigen::Vector3f::Zero())
    , m_gyro_bias(Eigen::Vector3f::Zero())
{
    // Get camera parameters from Config
    const Config& config = Config::getInstance();
    cv::Mat left_K = config.left_camera_matrix();
    
    if (!left_K.empty()) {
        m_fx = left_K.at<double>(0, 0);
        m_fy = left_K.at<double>(1, 1);
        m_cx = left_K.at<double>(0, 2);
        m_cy = left_K.at<double>(1, 2);
    } else {
        // Fallback to default values if config not available
        m_fx = 458.654; m_fy = 457.296; 
        m_cx = 367.215; m_cy = 248.375;
    }
    
    // Get distortion coefficients
    cv::Mat left_D = config.left_dist_coeffs();
    if (!left_D.empty()) {
        m_distortion_coeffs.clear();
        for (int i = 0; i < left_D.rows; ++i) {
            m_distortion_coeffs.push_back(left_D.at<double>(i, 0));
        }
    } else {
        m_distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0}; // No distortion
    }
    
    // Get T_BC (camera to body) from config and convert to T_CB (body to camera)
    cv::Mat T_bc_cv = config.left_T_BC();  // T_BC (camera to body)
    if (!T_bc_cv.empty()) {
        // Convert cv::Mat to Eigen::Matrix4d
        Eigen::Matrix4d T_bc;  // T_BC (camera to body)
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                T_bc(i, j) = T_bc_cv.at<double>(i, j);
            }
        }
        // Store T_CB = T_BC.inverse() → Convert to body to camera
        m_T_CB = T_bc.inverse();
    } else {
        // Fallback to identity if config not available
        m_T_CB = Eigen::Matrix4d::Identity();
    }
    
    // Set reference keyframe to last keyframe if available
    if (m_last_keyframe) {
        m_reference_keyframe = m_last_keyframe;
        m_T_relative_from_ref = Eigen::Matrix4f::Identity();
    }
    
    // Calculate undistorted border limits
    calculate_border();
}

void Frame::set_pose(const Eigen::Matrix3f& rotation, const Eigen::Vector3f& translation) {
    std::lock_guard<std::mutex> lock(m_pose_mutex);
    m_rotation = rotation;
    m_translation = translation;
}

void Frame::set_Twb(const Eigen::Matrix4f& T_wb) {
    std::lock_guard<std::mutex> lock(m_pose_mutex);
    m_rotation = T_wb.block<3, 3>(0, 0);
    m_translation = T_wb.block<3, 1>(0, 3);
}

Eigen::Matrix4f Frame::get_Twb() const {
    std::lock_guard<std::mutex> lock(m_pose_mutex);
    
    // If this frame is a keyframe, return its direct pose
    if (m_is_keyframe) {
        Eigen::Matrix4f T_wb = Eigen::Matrix4f::Identity();
        T_wb.block<3, 3>(0, 0) = m_rotation;
        T_wb.block<3, 1>(0, 3) = m_translation;
        return T_wb;
    }
    
    // For non-keyframes, try to get pose from reference keyframe
    auto ref_kf = m_reference_keyframe.lock();
    if (ref_kf) {
        // Get reference keyframe pose (could be gravity-transformed)
        Eigen::Matrix4f T_wb_ref = ref_kf->get_Twb();
        
        // Apply fixed relative transformation: T_wb = T_wb_ref * T_relative
        Eigen::Matrix4f T_wb = T_wb_ref * m_T_relative_from_ref;
        
        // Update internal pose storage for compatibility
        const_cast<Frame*>(this)->m_rotation = T_wb.block<3, 3>(0, 0);
        const_cast<Frame*>(this)->m_translation = T_wb.block<3, 1>(0, 3);
        
        return T_wb;
    } else {
        // Fallback to direct pose if no reference keyframe is available
        Eigen::Matrix4f T_wb = Eigen::Matrix4f::Identity();
        T_wb.block<3, 3>(0, 0) = m_rotation;
        T_wb.block<3, 1>(0, 3) = m_translation;
        return T_wb;
    }
}

Eigen::Matrix4f Frame::get_Twc() const {
    // Get T_wb (body to world transform) - this call is already thread-safe
    Eigen::Matrix4f T_wb = get_Twb();
    
    // Get T_CB (body to camera transform) 
    Eigen::Matrix4f T_CB = m_T_CB.cast<float>();
    
    // Calculate T_wc = T_wb * T_bc = T_wb * T_cb.inverse()
    Eigen::Matrix4f T_bc = T_CB.inverse();
    Eigen::Matrix4f T_wc = T_wb * T_bc;
    
    return T_wc;
}

void Frame::set_reference_keyframe(std::shared_ptr<Frame> reference_kf) {
    std::lock_guard<std::mutex> lock(m_pose_mutex);
    m_reference_keyframe = reference_kf;
    
    // Calculate relative transformation at the time of setting reference keyframe
    if (reference_kf) {
        // Current frame pose
        Eigen::Matrix4f T_wb_current = Eigen::Matrix4f::Identity();
        T_wb_current.block<3, 3>(0, 0) = m_rotation;
        T_wb_current.block<3, 1>(0, 3) = m_translation;
        
        // Reference keyframe pose
        Eigen::Matrix4f T_wb_ref = reference_kf->get_Twb();
        
        // Calculate relative transformation: T_rel = T_ref^-1 * T_current
        m_T_relative_from_ref = T_wb_ref.inverse() * T_wb_current;
    } else {
        // No reference keyframe, set to identity
        m_T_relative_from_ref = Eigen::Matrix4f::Identity();
    }
}

std::shared_ptr<Frame> Frame::get_reference_keyframe() const {
    return m_reference_keyframe.lock();
}

void Frame::add_feature(std::shared_ptr<Feature> feature) {
    m_features.push_back(feature);
    m_feature_id_to_index[feature->get_feature_id()] = m_features.size() - 1;
    // Add corresponding null map point and outlier flag
    m_map_points.push_back(nullptr);
    m_outlier_flags.push_back(false);
}

void Frame::remove_feature(int feature_id) {
    auto it = m_feature_id_to_index.find(feature_id);
    if (it != m_feature_id_to_index.end()) {
        size_t index = it->second;
        m_features.erase(m_features.begin() + index);
        m_map_points.erase(m_map_points.begin() + index);
        m_outlier_flags.erase(m_outlier_flags.begin() + index);
        m_feature_id_to_index.erase(it);
        update_feature_index();
    }
}

std::shared_ptr<Feature> Frame::get_feature(int feature_id) {
    auto it = m_feature_id_to_index.find(feature_id);
    if (it != m_feature_id_to_index.end()) {
        return m_features[it->second];
    }
    return nullptr;
}

int Frame::get_feature_index(int feature_id) const {
    auto it = m_feature_id_to_index.find(feature_id);
    if (it != m_feature_id_to_index.end()) {
        return static_cast<int>(it->second);
    }
    return -1;  // Feature not found
}

std::shared_ptr<const Feature> Frame::get_feature(int feature_id) const {
    auto it = m_feature_id_to_index.find(feature_id);
    if (it != m_feature_id_to_index.end()) {
        return m_features[it->second];
    }
    return nullptr;
}

void Frame::extract_features(int max_features) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (m_left_image.empty()) {
        std::cerr << "Cannot extract features: left image is empty" << std::endl;
        return;
    }

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(m_left_image, corners, max_features, m_quality_level, m_min_distance);

    // Use frame-local feature IDs starting from 0
    int local_feature_id = 0;
    for (const auto& corner : corners) {
        auto feature = std::make_shared<Feature>(local_feature_id++, corner);
        add_feature(feature);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Timing output removed for cleaner logs
}

cv::Mat Frame::draw_features() const {
    cv::Mat display_image;
    if (m_left_image.channels() == 1) {
        cv::cvtColor(m_left_image, display_image, cv::COLOR_GRAY2BGR);
    } else {
        display_image = m_left_image.clone();
    }

    for (size_t i = 0; i < m_features.size(); ++i) {
        const auto& feature = m_features[i];
        if (feature->is_valid()) {
            const cv::Point2f& pt = feature->get_pixel_coord();
            
            // Check if feature has associated map point
            auto map_point = get_map_point(i);
            cv::Scalar point_color;
            
            if (map_point && !map_point->is_bad()) {
                // Feature with MapPoint
                point_color = cv::Scalar(0, 0, 255);
            } else {
                // Feature without MapPoint
                point_color = cv::Scalar(255, 0, 0);
            }
            
            cv::circle(display_image, pt, 4, point_color, 2);
        }
    }

    return display_image;
}

cv::Mat Frame::draw_stereo_matches() const {
    if (!is_stereo()) {
        std::cout << "Cannot draw stereo matches: not a stereo frame" << std::endl;
        return cv::Mat();
    }

    // Create side-by-side display
    cv::Mat left_display, right_display;
    if (m_left_image.channels() == 1) {
        cv::cvtColor(m_left_image, left_display, cv::COLOR_GRAY2BGR);
        cv::cvtColor(m_right_image, right_display, cv::COLOR_GRAY2BGR);
    } else {
        left_display = m_left_image.clone();
        right_display = m_right_image.clone();
    }

    // Create combined image
    cv::Mat combined_image;
    cv::hconcat(left_display, right_display, combined_image);
    
    int right_offset = m_left_image.cols;

    // Draw features and matches
    for (const auto& feature : m_features) {
        if (feature->is_valid()) {
            const cv::Point2f& left_pt = feature->get_pixel_coord();
            
            // Draw left feature (green circle)
            cv::circle(combined_image, left_pt, 3, cv::Scalar(0, 255, 0), 2);
            
            if (feature->has_stereo_match()) {
                const cv::Point2f& right_pt = feature->get_right_coord();
                
                // Check if stereo match is valid (not (-1, -1))
                if (right_pt.x >= 0 && right_pt.y >= 0) {
                    cv::Point2f right_pt_shifted(right_pt.x + right_offset, right_pt.y);
                    
                    // Draw right feature (blue circle)
                    cv::circle(combined_image, right_pt_shifted, 3, cv::Scalar(255, 0, 0), 2);
                    
                    // Draw matching line (red)
                    cv::line(combined_image, left_pt, right_pt_shifted, cv::Scalar(0, 0, 255), 1);
                    
                    
                }
            }
        }
    }

    // Labels removed - text will be added by viewer instead

    return combined_image;
}

cv::Mat Frame::draw_rectified_stereo_matches() const {
    if (!is_stereo()) {
        std::cout << "Cannot draw normalized stereo matches: not a stereo frame" << std::endl;
        return cv::Mat();
    }

    // Create side-by-side display showing normalized coordinate space visualization
    cv::Mat left_display, right_display;
    if (m_left_image.channels() == 1) {
        cv::cvtColor(m_left_image, left_display, cv::COLOR_GRAY2BGR);
        cv::cvtColor(m_right_image, right_display, cv::COLOR_GRAY2BGR);
    } else {
        left_display = m_left_image.clone();
        right_display = m_right_image.clone();
    }

    // Create combined image
    cv::Mat combined_image;
    cv::hconcat(left_display, right_display, combined_image);
    
    int right_offset = m_left_image.cols;
    int valid_stereo_matches = 0;

    // Draw normalized features and matches
    for (const auto& feature : m_features) {
        if (feature->is_valid()) {
            // Use original pixel coordinates for visualization
            cv::Point2f left_px = feature->get_pixel_coord();
            
            // Draw left feature (green circle)
            cv::circle(combined_image, left_px, 3, cv::Scalar(0, 255, 0), 2);
            
            if (feature->has_stereo_match()) {
                cv::Point2f right_px = feature->get_right_coord();
                
                // Check if stereo match is valid
                if (right_px.x >= 0 && right_px.y >= 0) {
                    cv::Point2f right_px_shifted(right_px.x + right_offset, right_px.y);
                    
                    // Draw right feature (blue circle)
                    cv::circle(combined_image, right_px_shifted, 3, cv::Scalar(255, 0, 0), 2);
                    
                    // Draw matching line (pink/magenta)
                    cv::line(combined_image, left_px, right_px_shifted, cv::Scalar(255, 0, 255), 1);
                    
                    // Show 3D point depth if available
                    if (feature->has_3d_point()) {
                        Eigen::Vector3f pt3d = feature->get_3d_point();
                        std::string depth_str = cv::format("%.2fm", pt3d[2]);
                        cv::putText(combined_image, depth_str, 
                                   cv::Point(left_px.x + 5, left_px.y - 5), 
                                   cv::FONT_HERSHEY_SIMPLEX, 0.4, 
                                   cv::Scalar(255, 255, 255), 1);
                    }
                    
                    // Show normalized coordinates for debugging
                    Eigen::Vector2f norm = feature->get_normalized_coord();
                    if (norm[0] != 0.0f || norm[1] != 0.0f) { // Check if normalized coord is valid
                        std::string norm_str = cv::format("(%.2f,%.2f)", norm[0], norm[1]);
                        cv::putText(combined_image, norm_str, 
                                   cv::Point(left_px.x + 5, left_px.y + 15), 
                                   cv::FONT_HERSHEY_SIMPLEX, 0.3, 
                                   cv::Scalar(0, 255, 255), 1);
                    }
                    
                    valid_stereo_matches++;
                }
            }
        }
    }

    // Add labels
    cv::putText(combined_image, "Left (Normalized)", cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined_image, "Right (Normalized)", cv::Point(right_offset + 10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    // Show statistics
    cv::putText(combined_image, cv::format("Valid matches: %d", valid_stereo_matches), 
               cv::Point(10, combined_image.rows - 10), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    return combined_image;
}

cv::Mat Frame::draw_tracks(const Frame& previous_frame) const {
    cv::Mat display_image = draw_features();

    for (const auto& feature : m_features) {
        if (!feature->is_valid()) continue;

        // Use tracked_feature_id to find corresponding previous feature
        if (feature->has_tracked_feature()) {
            auto prev_feature = previous_frame.get_feature(feature->get_tracked_feature_id());
            if (prev_feature && prev_feature->is_valid()) {
                cv::line(display_image, 
                        prev_feature->get_pixel_coord(), 
                        feature->get_pixel_coord(), 
                        cv::Scalar(0, 255, 0), 1);
            }
        }
    }

    return display_image;
}

void Frame::update_feature_index() {
    m_feature_id_to_index.clear();
    for (size_t i = 0; i < m_features.size(); ++i) {
        m_feature_id_to_index[m_features[i]->get_feature_id()] = i;
    }
}

bool Frame::is_in_border(const cv::Point2f& point) const {
    // First undistort the input point to get its undistorted coordinates
    cv::Point2f undistorted_point = undistort_point(point);
    
    // Check against undistorted boundaries
    return (m_undist_x_min <= undistorted_point.x && 
            undistorted_point.x <= m_undist_x_max && 
            m_undist_y_min <= undistorted_point.y && 
            undistorted_point.y <= m_undist_y_max);
}

void Frame::compute_stereo_matches() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!is_stereo()) {
        std::cout << "Cannot compute stereo matches: right image not available" << std::endl;
        return;
    }

    std::vector<cv::Point2f> left_pts, right_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    // Extract feature points from left image
    for (const auto& feature : m_features) {
        if (feature->is_valid()) {
            left_pts.push_back(feature->get_pixel_coord());
        }
    }

    if (left_pts.empty()) {
        std::cout << "No features to match in stereo" << std::endl;
        return;
    }

    // Perform optical flow tracking from left to right image with stereo-specific parameters
    int stereo_window_size = Config::getInstance().m_stereo_window_size;
    cv::calcOpticalFlowPyrLK(m_left_image, m_right_image, left_pts, right_pts, 
                            status, err, cv::Size(stereo_window_size, stereo_window_size), 
                            Config::getInstance().m_stereo_max_level,
                            Config::getInstance().stereo_term_criteria(),
                            0, Config::getInstance().m_stereo_min_eigen_threshold); // Stereo-specific eigenvalue threshold

    int matches_found = 0;
    int optical_flow_failed = 0;
    int error_threshold_failed = 0;
    int epipolar_failed = 0;
    int y_diff_failed = 0;
    int disparity_failed = 0;
    int total_features = 0;
    
    // For unrectified stereo, we need more sophisticated matching
    // First, try to estimate fundamental matrix from initial matches
    std::vector<cv::Point2f> good_left_pts, good_right_pts;
    
    // Collect initial matches with very loose criteria
    size_t feature_idx = 0;
    for (auto& feature : m_features) {
        if (feature->is_valid() && feature_idx < status.size()) {
            if (status[feature_idx] && err[feature_idx] < Config::getInstance().m_stereo_error_threshold) { // Very loose error threshold
                good_left_pts.push_back(left_pts[feature_idx]);
                good_right_pts.push_back(right_pts[feature_idx]);
            }
            feature_idx++;
        }
    }
    
    cv::Mat fundamental_matrix;
    std::vector<uchar> inlier_mask;
    
   
    
    // Now apply matches with epipolar constraint
    feature_idx = 0;
    for (auto& feature : m_features) {
        if (feature->is_valid() && feature_idx < status.size()) {
            total_features++;
            
            if (!status[feature_idx]) {
                // Optical flow tracking failed
                optical_flow_failed++;
                feature->set_stereo_match(cv::Point2f(-1, -1), -1.0f);
                feature_idx++;
                continue;
            }
            
            if (err[feature_idx] >= Config::getInstance().m_stereo_error_threshold) {
                // Error threshold exceeded
                error_threshold_failed++;
                feature->set_stereo_match(cv::Point2f(-1, -1), -1.0f);
                feature_idx++;
                continue;
            }
            
            cv::Point2f left_pt = left_pts[feature_idx];
            cv::Point2f right_pt = right_pts[feature_idx];
            
            bool is_valid_match = true;
            
            // Check disparity (right point should be to the left of left point for positive disparity)
            float disparity = abs(left_pt.x - right_pt.x);
            if (disparity <= Config::getInstance().m_min_disparity || disparity >= Config::getInstance().m_max_disparity) {
                // Invalid disparity range
                disparity_failed++;
                is_valid_match = false;
            }
            
            // Check epipolar constraint if fundamental matrix is available
            if (is_valid_match && !fundamental_matrix.empty()) {
                // Convert points to homogeneous coordinates with correct type
                cv::Mat left_homo = (cv::Mat_<double>(3, 1) << left_pt.x, left_pt.y, 1.0);
                cv::Mat right_homo = (cv::Mat_<double>(3, 1) << right_pt.x, right_pt.y, 1.0);
                
                // Ensure fundamental matrix is double type
                cv::Mat F_double;
                if (fundamental_matrix.type() != CV_64F) {
                    fundamental_matrix.convertTo(F_double, CV_64F);
                } else {
                    F_double = fundamental_matrix;
                }
                
                // Compute epipolar error: x2^T * F * x1
                cv::Mat epipolar_error = right_homo.t() * F_double * left_homo;
                double error = std::abs(epipolar_error.at<double>(0, 0));
                
                // Reject if epipolar error is too large
                if (error > Config::getInstance().m_epipolar_threshold) {
                    epipolar_failed++;
                    is_valid_match = false;
                }
            }
            
            // Additional basic checks (remove disparity-based checks)
            if (is_valid_match) {
                float y_diff = std::abs(left_pt.y - right_pt.y);
                
                // Reject if y-coordinate difference is too large (basic sanity check)
                if (y_diff > Config::getInstance().m_max_y_difference) {
                    y_diff_failed++;
                    is_valid_match = false;
                }
            }
            
            if (is_valid_match) {
                feature->set_stereo_match(right_pt, -1.0f); // No disparity stored
                matches_found++;
            } else {
                // Invalid match - reset stereo match data
                feature->set_stereo_match(cv::Point2f(-1, -1), -1.0f);
            }
            
            feature_idx++;
        } else if (feature->is_valid()) {
            total_features++;
            // Feature is valid but no corresponding tracking result - set invalid disparity
            feature->set_stereo_match(cv::Point2f(-1, -1), -1.0f);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // STEREO debug logs removed - keeping only essential information
    float success_rate = total_features > 0 ? (float)matches_found / total_features * 100.0f : 0.0f;
    
    // // Debug stereo matching failures if significant
    // if (Config::getInstance().m_enable_debug_output && total_features > 0) {
    //     spdlog::info("[STEREO] Matching results: {}/{} successful ({:.1f}%)", 
    //                 matches_found, total_features, success_rate);
    //     spdlog::info("[STEREO] Failures: optical_flow={}, error_thresh={}, disparity={}, epipolar={}, y_diff={}", 
    //                 optical_flow_failed, error_threshold_failed, disparity_failed, epipolar_failed, y_diff_failed);
    // }
}

void Frame::undistort_features() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize outlier flags for all features
    initialize_outlier_flags();
    
    const Config& config = Config::getInstance();
    cv::Mat left_K = config.left_camera_matrix();
    cv::Mat left_D = config.left_dist_coeffs();
    cv::Mat right_K = config.right_camera_matrix();
    cv::Mat right_D = config.right_dist_coeffs();
    
    if (left_K.empty() || left_D.empty()) {
        std::cerr << "Camera calibration not available for undistortion" << std::endl;
        return;
    }
    
    // Convert to Eigen for easier computation  
    Eigen::Matrix3d K_left_eigen;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            K_left_eigen(i, j) = left_K.at<double>(i, j);
        }
    }
    
    // Process all features - convert to normalized coordinates only
    for (auto& feature : m_features) {
        if (feature->is_valid()) {
            // Get original pixel coordinate
            cv::Point2f pixel_pt = feature->get_pixel_coord();
            
            // Method 1: Direct undistortion to normalized coordinates (no rectification)
            std::vector<cv::Point2f> distorted_pts = {pixel_pt};
            std::vector<cv::Point2f> undistorted_pts;
            
            // Undistort to normalized coordinates based on camera model
            if (config.get_camera_model() == CameraModel::FISHEYE) {
                // Use fisheye undistortion
                cv::fisheye::undistortPoints(distorted_pts, undistorted_pts, left_K, left_D);
            } else {
                // Use standard pinhole undistortion (default)
                cv::undistortPoints(distorted_pts, undistorted_pts, left_K, left_D);
            }

            
            // Convert normalized coordinates back to pixel coordinates for set_undistorted_coord
            cv::Point2f undistorted_pixel;
            undistorted_pixel.x = undistorted_pts[0].x * m_fx + m_cx;
            undistorted_pixel.y = undistorted_pts[0].y * m_fy + m_cy;
            feature->set_undistorted_coord(undistorted_pixel);
            
            // Store normalized coordinate (from cv::undistortPoints output)
            Eigen::Vector2f normalized(undistorted_pts[0].x, undistorted_pts[0].y);
            feature->set_normalized_coord(normalized);
            
            // For stereo matches, we don't need rectification anymore
            // The triangulation will handle the geometric relationship directly
            if (feature->has_stereo_match()) {
                cv::Point2f right_pixel = feature->get_right_coord();
                
                // Check if stereo match is valid
                if (right_pixel.x >= 0 && right_pixel.y >= 0) {
                    // For right camera, undistort to normalized coordinates
                    std::vector<cv::Point2f> right_distorted = {right_pixel};
                    std::vector<cv::Point2f> right_undistorted;
                    
                    // For right camera, undistort to normalized coordinates based on camera model
                    if (config.get_camera_model() == CameraModel::FISHEYE) {
                        // Use fisheye undistortion
                        cv::fisheye::undistortPoints(right_distorted, right_undistorted, right_K, right_D);
                    } else {
                        // Use standard pinhole undistortion (default)
                        cv::undistortPoints(right_distorted, right_undistorted, right_K, right_D);
                    }
                    
                    // Also calculate normalized coordinates for right camera
                    Eigen::Vector2f right_normalized(right_undistorted[0].x, right_undistorted[0].y);
                    
                    // Convert normalized coordinates back to pixel coordinates for set_undistorted_stereo_match
                    cv::Point2f right_undistorted_pixel;
                    right_undistorted_pixel.x = right_undistorted[0].x * m_fx + m_cx; // Use left camera intrinsics for consistency
                    right_undistorted_pixel.y = right_undistorted[0].y * m_fy + m_cy;
                    
                    // Store right undistorted coordinate and normalized coordinate
                    feature->set_undistorted_stereo_match(right_undistorted_pixel, right_normalized, -1.0f);
                } else {
                    // Invalid stereo match
                    feature->set_undistorted_stereo_match(cv::Point2f(-1, -1), Eigen::Vector2f(-1,-1), -1.0f);
                }
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Count valid stereo matches (no disparity constraint)
    int valid_stereo_matches = 0;
    for (const auto& feature : m_features) {
        if (feature->is_valid() && feature->has_stereo_match()) {
            cv::Point2f right_undist = feature->get_right_undistorted_coord();
            if (right_undist.x >= 0 && right_undist.y >= 0) {
                valid_stereo_matches++;
            }
        }
    }
    
    // Timing output removed for cleaner logs
}

void Frame::triangulate_stereo_points() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const Config& config = Config::getInstance();
    cv::Mat left_K = config.left_camera_matrix();
    cv::Mat left_D = config.left_dist_coeffs();
    cv::Mat right_K = config.right_camera_matrix();
    cv::Mat right_D = config.right_dist_coeffs();
    cv::Mat T_rl = config.left_to_right_transform();  // T_rl: left to right transform (following T_ab = b->a convention)
    
    if (left_K.empty() || right_K.empty() || T_rl.empty()) {
        std::cerr << "Camera calibration not available for triangulation" << std::endl;
        return;
    }
    
    // Extract rotation and translation (T_rl: left to right transform)
    // This directly gives us left-to-right transformation for triangulation
    cv::Mat R_lr = T_rl(cv::Rect(0, 0, 3, 3));  // Left to right rotation  
    cv::Mat t_lr = T_rl(cv::Rect(3, 0, 1, 3));  // Left to right translation
    
    // Convert to Eigen for easier computation
    Eigen::Matrix3d R_eigen, K_left_eigen, K_right_eigen;
    Eigen::Vector3d t_eigen;
    
    // Convert CV matrices to Eigen (manual conversion)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R_eigen(i, j) = R_lr.at<double>(i, j);  // Use left-to-right rotation
            K_left_eigen(i, j) = left_K.at<double>(i, j);
            K_right_eigen(i, j) = right_K.at<double>(i, j);
        }
    }
    // Handle translation vector separately (3x1)
    t_eigen << t_lr.at<double>(0,0), t_lr.at<double>(1,0), t_lr.at<double>(2,0);  // Use left-to-right translation
    
    // Counters for debugging with detailed failure analysis
    int triangulated_count = 0;
    int depth_rejected = 0;
    int reprojection_rejected = 0;
    int total_stereo_matches = 0;
    
    // Detailed failure analysis counters
    int invalid_stereo_match_count = 0;
    int svd_fail_count = 0;
    int negative_depth_left = 0;
    int negative_depth_right = 0;
    
    // Detailed failure statistics
    std::vector<double> depth_values;
    std::vector<double> failed_depths;
    std::vector<double> reprojection_errors;
    std::vector<std::pair<std::string, int>> failure_reasons;
    
    for (auto& feature : m_features) {
        if (feature->is_valid() && feature->has_stereo_match()) {
            total_stereo_matches++;
            
            // Get pre-calculated normalized coordinates (more accurate)
            Eigen::Vector2f left_norm_2d = feature->get_normalized_coord();
            Eigen::Vector2f right_norm_2d = feature->get_right_normalized_coord();
            
            // Check if stereo match is valid
            if (right_norm_2d[0] == -1.0f || right_norm_2d[1] == -1.0f) {
                invalid_stereo_match_count++;
                if (config.m_enable_debug_output && invalid_stereo_match_count <= 5) {
                    cv::Point2f left_px = feature->get_pixel_coord();
                    cv::Point2f right_px = feature->get_right_coord();
                    // spdlog::debug("[TRIANGULATION] Invalid stereo match: left({:.1f},{:.1f}) -> right({:.1f},{:.1f})", 
                    //              left_px.x, left_px.y, right_px.x, right_px.y);
                }
                continue;
            }
            
            // Convert to 3D homogeneous coordinates for DLT
            Eigen::Vector3d left_normalized(left_norm_2d[0], left_norm_2d[1], 1.0);
            Eigen::Vector3d right_normalized(right_norm_2d[0], right_norm_2d[1], 1.0);
            
            // Triangulation using SVD
            // Setup design matrix: A * X = 0
            Eigen::Matrix4d A;
            
            // Left camera projection matrix is [I | 0] (identity pose)
            Eigen::Matrix<double, 3, 4> P_left;
            P_left.setZero();
            P_left.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
            
            // Right camera projection matrix using T_rl (left-to-right transform)
            // T_rl directly gives us left-to-right transformation
            Eigen::Matrix<double, 3, 4> P_right;
            P_right.block<3,3>(0,0) = R_eigen;  // R_lr (left-to-right rotation)
            P_right.block<3,1>(0,3) = t_eigen;  // t_lr (left-to-right translation)
            
            // Build constraint equations: normalized_point^T * [P * X] = 0
            A.row(0) = left_normalized[0] * P_left.row(2) - P_left.row(0);
            A.row(1) = left_normalized[1] * P_left.row(2) - P_left.row(1);
            A.row(2) = right_normalized[0] * P_right.row(2) - P_right.row(0);
            A.row(3) = right_normalized[1] * P_right.row(2) - P_right.row(1);
            
            // Solve using SVD
            Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
            
            Eigen::Vector4d X_h = svd.matrixV().col(3);
            
            // Check homogeneous coordinate
            if (std::abs(X_h[3]) < 1e-3) {
                svd_fail_count++;
                if (config.m_enable_debug_output && svd_fail_count <= 5) {
                    // spdlog::debug("[TRIANGULATION] SVD failed: homogeneous coordinate too small ({:.2e})", X_h[3]);
                    // spdlog::debug("  Left norm: ({:.3f},{:.3f}), Right norm: ({:.3f},{:.3f})", 
                    //              left_norm_2d[0], left_norm_2d[1], right_norm_2d[0], right_norm_2d[1]);
                }
                continue;
            }
            
            // Convert to 3D point in left camera frame
            X_h /= X_h[3];
            Eigen::Vector3d pos_3d = X_h.head(3);
            
            // Detailed depth analysis
            if (pos_3d[2] <= 0) {
                negative_depth_left++;
                if (config.m_enable_debug_output && negative_depth_left <= 3) {
                    // spdlog::debug("[TRIANGULATION] Negative depth in left camera: {:.3f}", pos_3d[2]);
                }
                failed_depths.push_back(pos_3d[2]);
                continue;
            }
            
            depth_values.push_back(pos_3d[2]);
            
            // Debug: Print depth values to see what we're getting
            if (config.m_enable_debug_output && depth_values.size() <= 10) {
                // std::cout << "Triangulated depth: " << pos_3d[2] 
                //           << ", 3D point: (" << pos_3d[0] << ", " << pos_3d[1] << ", " << pos_3d[2] << ")" 
                //           << ", left_norm: (" << left_norm_2d[0] << ", " << left_norm_2d[1] << ")"
                //           << ", right_norm: (" << right_norm_2d[0] << ", " << right_norm_2d[1] << ")" << std::endl;
            }
            
            // Check depth range (positive depth in front of camera)
            if (pos_3d[2] < config.m_min_depth || pos_3d[2] > config.m_max_depth) {
                depth_rejected++;
                if (config.m_enable_debug_output && depth_rejected <= 10) {
                    bool too_close = pos_3d[2] < config.m_min_depth;
                    bool too_far = pos_3d[2] > config.m_max_depth;
                    SPDLOG_DEBUG("Depth range violation: {:.3f}m ({}) at pixel ({:.1f},{:.1f}) - range [{:.1f},{:.1f}]m",
                                pos_3d[2], 
                                too_close ? "too close" : "too far",
                                stereo_matches[i].first.x, stereo_matches[i].first.y,
                                config.m_min_depth, config.m_max_depth);
                }
                continue;
            }
            
            // Reprojection error check using normalized coordinates
            // Project 3D point back to left camera (should match left_normalized)
            Eigen::Vector3d reproj_left = pos_3d; // Already in left camera frame
            reproj_left /= reproj_left[2]; // Normalize by depth
            
            // Project 3D point to right camera using T_rl (left-to-right)
            Eigen::Vector3d pos_right = R_eigen * pos_3d + t_eigen;
            if (pos_right[2] <= 0) { // Check positive depth in right camera
                negative_depth_right++;
                if (config.m_enable_debug_output && negative_depth_right <= 10) {
                    SPDLOG_DEBUG("Negative depth in right camera: {:.3f} at pixel ({:.1f},{:.1f})",
                                pos_right[2], 
                                stereo_matches[i].first.x, stereo_matches[i].first.y);
                }
                continue;
            }
            Eigen::Vector3d reproj_right = pos_right / pos_right[2];
            
            // Calculate reprojection errors in normalized coordinates
            double left_error = (left_normalized.head<2>() - reproj_left.head<2>()).norm();
            double right_error = (right_normalized.head<2>() - reproj_right.head<2>()).norm();
            double max_error = std::max(left_error, right_error);
            
            // Convert reprojection error to pixel coordinates for storage
            double max_error_pixels = max_error * std::min(K_left_eigen(0,0), K_left_eigen(1,1));
            
            // Use normalized coordinate reprojection threshold
            double max_reproj_error = sqrt(5.991) / std::min(K_left_eigen(0,0), K_left_eigen(1,1));
            
            if (max_error > max_reproj_error) {
                reprojection_rejected++;
                if (config.m_enable_debug_output && reprojection_rejected <= 10) {
                    SPDLOG_DEBUG("Reprojection error too high: L={:.6f}, R={:.6f}, max={:.6f}, thresh={:.6f} at pixel ({:.1f},{:.1f})",
                                left_error, right_error, max_error, max_reproj_error,
                                stereo_matches[i].first.x, stereo_matches[i].first.y);
                    SPDLOG_DEBUG("  Expected: L=({:.6f},{:.6f}), R=({:.6f},{:.6f})",
                                left_normalized[0], left_normalized[1],
                                right_normalized[0], right_normalized[1]);
                    SPDLOG_DEBUG("  Reprojected: L=({:.6f},{:.6f}), R=({:.6f},{:.6f})",
                                reproj_left[0], reproj_left[1],
                                reproj_right[0], reproj_right[1]);
                }
                continue;
            }
            
            // Success! Store the 3D point in left camera frame (as per Feature.h comment)
            Eigen::Vector3f point3D_camera = pos_3d.cast<float>();
            
            // Store in camera coordinates - transformation to body/world will be done in Estimator
            feature->set_3d_point(point3D_camera);
            
            // Store reprojection error from triangulation (in pixel coordinates)
            feature->set_reprojection_error(static_cast<float>(max_error_pixels));
            
            // Update normalized coordinates in feature (using left camera)
            Eigen::Vector2f normalized_2d(left_normalized[0], left_normalized[1]);
            feature->set_normalized_coord(normalized_2d);
            
            triangulated_count++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Detailed triangulation failure analysis
    int total_failures = invalid_stereo_match_count + svd_fail_count + negative_depth_left + 
                        negative_depth_right + depth_rejected + reprojection_rejected;
    
    // // Log triangulation results for debugging
    // spdlog::info("[TRIANGULATION] Frame {}: {}/{} features triangulated ({:.1f}%) in {:.1f}ms", 
    //             m_frame_id, triangulated_count, total_stereo_matches, 
    //             (triangulated_count * 100.0) / std::max(1, total_stereo_matches),
    //             duration.count() / 1000.0);
    
    // Always show failure breakdown if there are failures
    // if (total_failures > 0) {
    //     // spdlog::info("Failure breakdown: invalid_stereo={}, svd_fail={}, neg_depth_L={}, neg_depth_R={}, depth_range={}, reproj_error={}",
    //     //             invalid_stereo_match_count, svd_fail_count, negative_depth_left, 
    //     //             negative_depth_right, depth_rejected, reprojection_rejected);
        
    //     // if (config.m_enable_debug_output) {
    //     //     double invalid_rate = (invalid_stereo_match_count * 100.0) / std::max(1, total_stereo_matches);
    //     //     double svd_rate = (svd_fail_count * 100.0) / std::max(1, total_stereo_matches);
    //     //     double depth_l_rate = (negative_depth_left * 100.0) / std::max(1, total_stereo_matches);
    //     //     double depth_r_rate = (negative_depth_right * 100.0) / std::max(1, total_stereo_matches);
    //     //     double depth_range_rate = (depth_rejected * 100.0) / std::max(1, total_stereo_matches);
    //     //     double reproj_rate = (reprojection_rejected * 100.0) / std::max(1, total_stereo_matches);
            
    //     //     spdlog::info("Failure rates: invalid={:.1f}%, svd={:.1f}%, neg_L={:.1f}%, neg_R={:.1f}%, range={:.1f}%, reproj={:.1f}%",
    //     //                 invalid_rate, svd_rate, depth_l_rate, depth_r_rate, depth_range_rate, reproj_rate);
    //     // }
    // }
}

void Frame::initialize_map_points() {
    m_map_points.clear();
    m_map_points.resize(m_features.size(), nullptr);
}

void Frame::set_map_point(int feature_index, std::shared_ptr<MapPoint> map_point) {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_map_points.size())) {
        m_map_points[feature_index] = map_point;
    }
}

std::shared_ptr<MapPoint> Frame::get_map_point(int feature_index) const {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_map_points.size())) {
        return m_map_points[feature_index];
    }
    return nullptr;
}

bool Frame::has_map_point(int feature_index) const {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_map_points.size())) {
        return m_map_points[feature_index] != nullptr;
    }
    return false;
}

// Outlier flag management
void Frame::set_outlier_flag(int feature_index, bool is_outlier) {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_outlier_flags.size())) {
        m_outlier_flags[feature_index] = is_outlier;
    }
}

bool Frame::get_outlier_flag(int feature_index) const {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_outlier_flags.size())) {
        return m_outlier_flags[feature_index];
    }
    return false; // Default to not outlier if index is invalid
}

void Frame::initialize_outlier_flags() {
    m_outlier_flags.assign(m_features.size(), false);
}

// Camera parameter management
void Frame::set_camera_intrinsics(double fx, double fy, double cx, double cy) {
    m_fx = fx;
    m_fy = fy;
    m_cx = cx;
    m_cy = cy;
}

void Frame::get_camera_intrinsics(double& fx, double& fy, double& cx, double& cy) const {
    fx = m_fx;
    fy = m_fy;
    cx = m_cx;
    cy = m_cy;
}

void Frame::set_distortion_coeffs(const std::vector<double>& distortion_coeffs) {
    m_distortion_coeffs = distortion_coeffs;
}

cv::Point2f Frame::undistort_point(const cv::Point2f& distorted_point) const {
    if (m_distortion_coeffs.empty() || 
        (m_distortion_coeffs.size() >= 5 && 
         std::abs(m_distortion_coeffs[0]) < 1e-6 && 
         std::abs(m_distortion_coeffs[1]) < 1e-6)) {
        return distorted_point; // No significant distortion correction needed
    }

    // Convert to normalized coordinates
    double x = (distorted_point.x - m_cx) / m_fx;
    double y = (distorted_point.y - m_cy) / m_fy;

    // Iterative undistortion (Newton-Raphson method)
    if (m_distortion_coeffs.size() >= 5) {
        double k1 = m_distortion_coeffs[0];
        double k2 = m_distortion_coeffs[1];
        double p1 = m_distortion_coeffs[2];
        double p2 = m_distortion_coeffs[3];
        double k3 = m_distortion_coeffs[4];

        // Initial guess
        double x_u = x;
        double y_u = y;

        // Iterative correction (typically 5 iterations are enough)
        for (int iter = 0; iter < 5; ++iter) {
            double r2 = x_u*x_u + y_u*y_u;
            double r4 = r2*r2;
            double r6 = r4*r2;

            // Radial distortion
            double radial_factor = 1.0 + k1*r2 + k2*r4 + k3*r6;
            
            // Tangential distortion
            double dx = 2.0*p1*x_u*y_u + p2*(r2 + 2.0*x_u*x_u);
            double dy = p1*(r2 + 2.0*y_u*y_u) + 2.0*p2*x_u*y_u;

            // Distorted coordinates
            double x_d = x_u * radial_factor + dx;
            double y_d = y_u * radial_factor + dy;

            // Correction
            x_u = x_u - (x_d - x);
            y_u = y_u - (y_d - y);
        }

        x = x_u;
        y = y_u;
    }

    // Convert back to pixel coordinates
    return cv::Point2f(x * m_fx + m_cx, y * m_fy + m_cy);
}

void Frame::extract_stereo_features(int max_features) {
    // Extract features only from left image
    extract_features(max_features);
    
    // Initialize stereo match vectors
    m_stereo_matches.assign(m_features.size(), -1);
    m_depths.assign(m_features.size(), -1.0);
}

void Frame::compute_stereo_depth() {
    // First compute stereo matches
    compute_stereo_matches();
    
    // Then undistort features
    undistort_features();
    
    // Finally triangulate to get 3D points and extract depth
    triangulate_stereo_points();
    
    // Update depth array from triangulated 3D points
    m_depths.assign(m_features.size(), -1.0);
    
    for (size_t i = 0; i < m_features.size(); ++i) {
        auto feature = m_features[i];
        if (feature && feature->is_valid() && feature->has_3d_point()) {
            Eigen::Vector3f point3d = feature->get_3d_point();
            m_depths[i] = point3d[2]; // Z coordinate is depth in camera frame
        }
    }
}

double Frame::get_depth(int feature_index) const {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_depths.size())) {
        return m_depths[feature_index];
    }
    return -1.0;
}

bool Frame::has_depth(int feature_index) const {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_depths.size())) {
        return m_depths[feature_index] > 0.0;
    }
    return false;
}

bool Frame::has_valid_stereo_depth(const cv::Point2f& pixel_coord) const {
    // Check if pixel coordinate is within image bounds
    if (pixel_coord.x < 0 || pixel_coord.y < 0 || 
        pixel_coord.x >= m_left_image.cols || pixel_coord.y >= m_left_image.rows) {
        return false;
    }
    
    // Compute stereo disparity and check if valid
    double disparity = compute_disparity_at_point(pixel_coord);
    return disparity > 0.0;
}


double Frame::compute_disparity_at_point(const cv::Point2f& pixel_coord) const {
    // Simple stereo matching using normalized cross correlation
    int x = static_cast<int>(pixel_coord.x);
    int y = static_cast<int>(pixel_coord.y);
    
    if (x < 0 || y < 0 || x >= m_left_image.cols || y >= m_left_image.rows) {
        return 0.0;
    }
    
    // Search window parameters
    int search_range = 64;  // Maximum disparity to search
    int window_size = 5;    // Correlation window size
    int half_window = window_size / 2;
    
    // Check if we have enough space for correlation window
    if (x - half_window < 0 || x + half_window >= m_left_image.cols ||
        y - half_window < 0 || y + half_window >= m_left_image.rows) {
        return 0.0;
    }
    
    double best_disparity = 0.0;
    double best_correlation = -1.0;
    
    // Extract left patch
    cv::Rect left_rect(x - half_window, y - half_window, window_size, window_size);
    cv::Mat left_patch = m_left_image(left_rect);
    
    // Search along epipolar line (assuming rectified stereo)
    for (int d = 1; d < search_range && (x - d) >= half_window; ++d) {
        cv::Rect right_rect(x - d - half_window, y - half_window, window_size, window_size);
        
        if (right_rect.x >= 0 && right_rect.x + right_rect.width <= m_right_image.cols) {
            cv::Mat right_patch = m_right_image(right_rect);
            
            // Compute normalized cross correlation
            cv::Mat correlation_result;
            cv::matchTemplate(left_patch, right_patch, correlation_result, cv::TM_CCOEFF_NORMED);
            
            double correlation = correlation_result.at<float>(0, 0);
            
            if (correlation > best_correlation) {
                best_correlation = correlation;
                best_disparity = static_cast<double>(d);
            }
        }
    }
    
    // Only accept if correlation is strong enough
    if (best_correlation > 0.7) {
        return best_disparity;
    }
    
    return 0.0;
}

// IMU data management methods
void Frame::set_imu_data_from_last_frame(const std::vector<IMUData>& imu_data) {
    m_imu_vec_from_last_frame = imu_data;
}

void Frame::set_imu_data_since_last_keyframe(const std::vector<IMUData>& imu_data) {
    m_imu_vec_since_last_keyframe = imu_data;
}

void Frame::set_imu_preintegration_from_last_keyframe(std::shared_ptr<IMUPreintegration> preintegration) {
    m_imu_preintegration_from_last_keyframe = preintegration;
}

void Frame::set_imu_preintegration_from_last_frame(std::shared_ptr<IMUPreintegration> preintegration) {
    m_imu_preintegration_from_last_frame = preintegration;
}

// VIO-specific pose and velocity methods
Sophus::SE3f Frame::get_world_pose() const {
    return m_world_pose;
}

void Frame::set_world_pose(const Sophus::SE3f& pose) {
    m_world_pose = pose;
}

void Frame::initialize_velocity_from_preintegration() {
    // If we already have non-zero velocity, keep it
    if (m_velocity.norm() > 1e-6) {
        return;
    }


    // spdlog::warn("Check this function call - Frame::initialize_velocity_from_preintegration()");
    
    std::vector<Eigen::Vector3f> velocity_candidates;
    std::vector<std::string> velocity_sources;
    
    // ===============================================================================
    // METHOD 1: From last keyframe preintegration
    // ===============================================================================
    auto preint_from_keyframe = get_imu_preintegration_from_last_keyframe();
    double dt_from_keyframe = get_dt_from_last_keyframe();
    
    if (preint_from_keyframe && dt_from_keyframe > 0.001 && dt_from_keyframe < 1.0) {
        auto last_keyframe = get_last_keyframe();
        if (last_keyframe) {
            // Transform preintegrated velocity to world frame using last keyframe rotation
            Eigen::Matrix4f T_wb_last = last_keyframe->get_Twb();
            Eigen::Matrix3f R_wb_last = T_wb_last.block<3,3>(0,0);
            
            // Compute velocity in world frame from keyframe preintegration (delta_V is velocity, don't divide by time!)
            Eigen::Vector3f velocity_from_keyframe = R_wb_last * preint_from_keyframe->delta_V;
            
            velocity_candidates.push_back(velocity_from_keyframe);
            velocity_sources.push_back("from_keyframe");
            
        }
    }
    
    // ===============================================================================
    // METHOD 2: From last frame preintegration (if available)
    // ===============================================================================
    auto preint_from_frame = get_imu_preintegration_from_last_frame();
    
    if (preint_from_frame && preint_from_frame->dt_total > 0.001 && preint_from_frame->dt_total < 1.0) {
        // For frame-to-frame preintegration, we need the previous frame's pose
        // Since we don't have direct access to previous frame, we can use current frame's pose
        // as approximation for very short time intervals
        
        Eigen::Matrix4f T_wb_current = get_Twb();
        Eigen::Matrix3f R_wb_current = T_wb_current.block<3,3>(0,0);
        
        // Compute velocity from frame preintegration (delta_V is already velocity, don't divide by time!)
        Eigen::Vector3f velocity_from_frame = R_wb_current * preint_from_frame->delta_V;
        
        velocity_candidates.push_back(velocity_from_frame);
        velocity_sources.push_back("from_frame");
        
    }
    
    // ===============================================================================
    // COMPUTE AVERAGE VELOCITY AND SET
    // ===============================================================================
    if (velocity_candidates.empty()) {
        // No preintegration data available, keep zero velocity
        return;
    }
    
    // Compute average velocity from all available sources
    Eigen::Vector3f average_velocity = Eigen::Vector3f::Zero();
    for (const auto& vel : velocity_candidates) {
        average_velocity += vel;
    }
    average_velocity /= static_cast<float>(velocity_candidates.size());
    
    // Set the averaged velocity
    set_velocity(average_velocity);
    
    // Create sources string for logging
    std::string sources_str = "";
    for (size_t i = 0; i < velocity_sources.size(); ++i) {
        sources_str += velocity_sources[i];
        if (i < velocity_sources.size() - 1) sources_str += "+";
    }
    
}

void Frame::calculate_border() {
    // Get image dimensions
    int img_width = m_left_image.cols;
    int img_height = m_left_image.rows;
    
    // Define the four corner points of the image
    std::vector<cv::Point2f> corner_points = {
        cv::Point2f(0, 0),                           // Top-left
        cv::Point2f(img_width - 1, 0),               // Top-right  
        cv::Point2f(0, img_height - 1),              // Bottom-left
        cv::Point2f(img_width - 1, img_height - 1)   // Bottom-right
    };
    
    undistort_corner_points(corner_points);

    // spdlog::info("[BORDER] Frame {}: Undistorted borders - X:[{:.1f}, {:.1f}], Y:[{:.1f}, {:.1f}]", 
    //             m_frame_id, m_undist_x_min, m_undist_x_max, m_undist_y_min, m_undist_y_max);
}

void Frame::undistort_corner_points(const std::vector<cv::Point2f>& corner_points) {
    const Config& config = Config::getInstance();
    cv::Mat left_K = config.left_camera_matrix();
    cv::Mat left_D = config.left_dist_coeffs();
    
    if (left_K.empty() || left_D.empty()) {
        std::cerr << "Camera calibration not available for border calculation" << std::endl;
        // Fallback to simple pixel boundaries
        m_undist_x_min = 0;
        m_undist_x_max = m_left_image.cols - 1;
        m_undist_y_min = 0; 
        m_undist_y_max = m_left_image.rows - 1;
        return;
    }
    
    // Undistort corner points to normalized coordinates
    std::vector<cv::Point2f> undistorted_corners;
    
    if (config.get_camera_model() == CameraModel::FISHEYE) {
        // Use fisheye undistortion
        cv::fisheye::undistortPoints(corner_points, undistorted_corners, left_K, left_D);
    } else {
        // Use standard pinhole undistortion (default)
        cv::undistortPoints(corner_points, undistorted_corners, left_K, left_D);
    }
    
    // Convert back to pixel coordinates to get undistorted boundary
    std::vector<cv::Point2f> undistorted_pixels;
    for (const auto& norm_pt : undistorted_corners) {
        cv::Point2f pixel_pt;
        pixel_pt.x = norm_pt.x * m_fx + m_cx;
        pixel_pt.y = norm_pt.y * m_fy + m_cy;
        undistorted_pixels.push_back(pixel_pt);
    }
    
    // Find min/max bounds from all undistorted corner points
    m_undist_x_min = undistorted_pixels[0].x;
    m_undist_x_max = undistorted_pixels[0].x;
    m_undist_y_min = undistorted_pixels[0].y;
    m_undist_y_max = undistorted_pixels[0].y;
    
    for (const auto& pt : undistorted_pixels) {
        m_undist_x_min = std::min(m_undist_x_min, static_cast<double>(pt.x));
        m_undist_x_max = std::max(m_undist_x_max, static_cast<double>(pt.x));
        m_undist_y_min = std::min(m_undist_y_min, static_cast<double>(pt.y));
        m_undist_y_max = std::max(m_undist_y_max, static_cast<double>(pt.y));
    }
    
    // Debug output for border calculation
    const Config& config_debug = Config::getInstance();
    if (config_debug.m_enable_debug_output) {
        spdlog::info("[BORDER] Frame {}: Undistorted borders - X:[{:.1f}, {:.1f}], Y:[{:.1f}, {:.1f}]", 
                    m_frame_id, m_undist_x_min, m_undist_x_max, m_undist_y_min, m_undist_y_max);
        
        // Show original vs undistorted corner coordinates
        for (size_t i = 0; i < corner_points.size(); ++i) {
            spdlog::debug("[BORDER] Corner {}: ({:.1f},{:.1f}) -> normalized:({:.3f},{:.3f}) -> undist_pixel:({:.1f},{:.1f})", 
                         i, corner_points[i].x, corner_points[i].y,
                         undistorted_corners[i].x, undistorted_corners[i].y,
                         undistorted_pixels[i].x, undistorted_pixels[i].y);
        }
    }
}
} // namespace lightweight_vio
