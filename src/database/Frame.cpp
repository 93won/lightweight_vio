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
#include "camera/Camera.h"
#include "camera/Rectlinear.h"
#include "camera/Fisheye.h"
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

void Frame::release_images() {
    // Release all heavy image data
    if (!m_left_image.empty()) {
        m_left_image = cv::Mat();
        m_left_image.release();
    }
    if (!m_right_image.empty()) {
        m_right_image = cv::Mat();
        m_right_image.release();
    }
    if (!m_rgb_image.empty()) {
        m_rgb_image = cv::Mat();
        m_rgb_image.release();
    }
    if (!m_depth_map.empty()) {
        m_depth_map = cv::Mat();
        m_depth_map.release();
    }
    

}

// ============================================================================
// NEW CONSTRUCTORS WITH CAMERA CLASS
// ============================================================================

Frame::Frame(double timestamp, int frame_id, std::shared_ptr<Camera> camera)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_frame_type(FrameType::STEREO)
    , m_camera(camera)
    , m_rotation(Eigen::Matrix3f::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_is_keyframe(false)
    , m_is_active(true)
    , m_world_pose(Sophus::SE3f())
    , m_velocity(Eigen::Vector3f::Zero())
    , m_accel_bias(Eigen::Vector3f::Zero())
    , m_gyro_bias(Eigen::Vector3f::Zero())
    , m_dt_from_last_keyframe(0.0)
    , m_T_relative_from_ref(Eigen::Matrix4f::Identity())
    , m_fx(camera->get_fx()), m_fy(camera->get_fy())
    , m_cx(camera->get_cx()), m_cy(camera->get_cy())
    , m_distortion_coeffs(camera->get_distortion_coeffs())
{
    // Get T_BC from config and convert to T_CB (body to camera)
    const Config& config = Config::getInstance();
    cv::Mat T_bc_cv = config.left_T_BC();
    Eigen::Matrix4d T_bc;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T_bc(i, j) = T_bc_cv.at<double>(i, j);
        }
    }
    m_T_CB = T_bc.inverse();
    
    // Set reference keyframe to last keyframe if available
    if (m_last_keyframe) {
        m_reference_keyframe = m_last_keyframe;
        m_T_relative_from_ref = Eigen::Matrix4f::Identity();
    }
}

// Monocular constructor with image
Frame::Frame(double timestamp, int frame_id,
             const cv::Mat& image,
             std::shared_ptr<Camera> camera)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_frame_type(FrameType::MONOCULAR)
    , m_camera(camera)
    , m_right_camera(nullptr)
{
    m_left_image = image.clone();
    
    m_rotation = Eigen::Matrix3f::Identity();
    m_translation = Eigen::Vector3f::Zero();
    m_is_keyframe = false;
    m_is_active = true;
    m_world_pose = Sophus::SE3f();
    m_velocity = Eigen::Vector3f::Zero();
    m_accel_bias = Eigen::Vector3f::Zero();
    m_gyro_bias = Eigen::Vector3f::Zero();
    m_dt_from_last_keyframe = 0.0;
    m_T_relative_from_ref = Eigen::Matrix4f::Identity();
    
    m_fx = camera->get_fx();
    m_fy = camera->get_fy();
    m_cx = camera->get_cx();
    m_cy = camera->get_cy();
    m_distortion_coeffs = camera->get_distortion_coeffs();
    
    // Get T_BC from config and convert to T_CB (body to camera)
    const Config& config = Config::getInstance();
    cv::Mat T_bc_cv = config.left_T_BC();
    
    Eigen::Matrix4d T_bc;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T_bc(i, j) = T_bc_cv.at<double>(i, j);
        }
    }
    m_T_CB = T_bc.inverse();
    
    // Set reference keyframe to last keyframe if available
    if (m_last_keyframe) {
        m_reference_keyframe = m_last_keyframe;
        m_T_relative_from_ref = Eigen::Matrix4f::Identity();
    }
    
    // Compute undistorted image boundaries
    undistort_corner_points();
}

Frame::Frame(double timestamp, int frame_id,
             const cv::Mat& left_image, const cv::Mat& right_image,
             std::shared_ptr<Camera> left_camera, std::shared_ptr<Camera> right_camera)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_frame_type(FrameType::STEREO)
    , m_camera(left_camera)
    , m_right_camera(right_camera)
    , m_left_image(left_image.clone())
    , m_right_image(right_image.clone())
    , m_rotation(Eigen::Matrix3f::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_is_keyframe(false)
    , m_is_active(true)
    , m_world_pose(Sophus::SE3f())
    , m_velocity(Eigen::Vector3f::Zero())
    , m_accel_bias(Eigen::Vector3f::Zero())
    , m_gyro_bias(Eigen::Vector3f::Zero())
    , m_dt_from_last_keyframe(0.0)
    , m_T_relative_from_ref(Eigen::Matrix4f::Identity())
    , m_fx(left_camera->get_fx()), m_fy(left_camera->get_fy())
    , m_cx(left_camera->get_cx()), m_cy(left_camera->get_cy())
    , m_distortion_coeffs(left_camera->get_distortion_coeffs())
{
    // Get T_BC from config and convert to T_CB (body to camera)
    const Config& config = Config::getInstance();
    cv::Mat T_bc_cv = config.left_T_BC();
    Eigen::Matrix4d T_bc;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T_bc(i, j) = T_bc_cv.at<double>(i, j);
        }
    }
    m_T_CB = T_bc.inverse();
    
    // Set reference keyframe to last keyframe if available
    if (m_last_keyframe) {
        m_reference_keyframe = m_last_keyframe;
        m_T_relative_from_ref = Eigen::Matrix4f::Identity();
    }
    
    // Compute undistorted image boundaries
    undistort_corner_points();
}

Frame::Frame(double timestamp, int frame_id,
             const cv::Mat &rgb_image, const cv::Mat &depth_map,
             std::shared_ptr<Camera> camera, bool is_rgbd)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_frame_type(FrameType::RGBD)
    , m_camera(camera)
    , m_right_camera(nullptr)  // RGBD has no right camera
    , m_left_image(rgb_image.clone())
    , m_rotation(Eigen::Matrix3f::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_is_keyframe(false)
    , m_is_active(true)
    , m_world_pose(Sophus::SE3f())
    , m_velocity(Eigen::Vector3f::Zero())
    , m_accel_bias(Eigen::Vector3f::Zero())
    , m_gyro_bias(Eigen::Vector3f::Zero())
    , m_dt_from_last_keyframe(0.0)
    , m_T_relative_from_ref(Eigen::Matrix4f::Identity())
    , m_depth_map(depth_map.clone())
    , m_fx(camera->get_fx()), m_fy(camera->get_fy())
    , m_cx(camera->get_cx()), m_cy(camera->get_cy())
    , m_distortion_coeffs(camera->get_distortion_coeffs())
{
    // Get T_BC from config and convert to T_CB (body to camera)
    const Config& config = Config::getInstance();
    cv::Mat T_bc_cv = config.left_T_BC();
    Eigen::Matrix4d T_bc;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T_bc(i, j) = T_bc_cv.at<double>(i, j);
        }
    }
    m_T_CB = T_bc.inverse();
    
    // Set reference keyframe to last keyframe if available
    if (m_last_keyframe) {
        m_reference_keyframe = m_last_keyframe;
        m_T_relative_from_ref = Eigen::Matrix4f::Identity();
    }
    
    // Process depth map
    process_depth_map(depth_map);
    
    // Compute undistorted image boundaries
    undistort_corner_points();
}

// ============================================================================
// DEPRECATED CONSTRUCTORS (for backward compatibility)
// ============================================================================

Frame::Frame(double timestamp, int frame_id, 
             double fx, double fy, double cx, double cy, 
             const std::vector<double>& distortion_coeffs)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_frame_type(FrameType::STEREO)  // ⭐ Default to STEREO
    , m_rotation(Eigen::Matrix3f::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_is_keyframe(false)
    , m_is_active(true)
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
}

Frame::Frame(double timestamp, int frame_id,
             const cv::Mat& left_image, const cv::Mat& right_image,
             double fx, double fy, double cx, double cy, 
             const std::vector<double>& distortion_coeffs)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_frame_type(FrameType::STEREO)  // ⭐ Stereo frame
    , m_left_image(left_image.clone())
    , m_right_image(right_image.clone())
    , m_rotation(Eigen::Matrix3f::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_is_keyframe(false)
    , m_is_active(true)
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
    
    // Compute undistorted image boundaries
    undistort_corner_points();
}


// RGBD constructor - directly takes both images, uses Config for camera params
Frame::Frame(double timestamp, int frame_id,
             const cv::Mat &rgb_image, const cv::Mat &depth_map,
             double fx, double fy, double cx, double cy, const std::vector<double> &distortion_coeffs, bool is_rgbd)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_frame_type(FrameType::RGBD)  // ⭐ RGBD frame
    , m_left_image(rgb_image.clone())  // Store RGB as left image
    , m_rotation(Eigen::Matrix3f::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_is_keyframe(false)
    , m_is_active(true)
    , m_world_pose(Sophus::SE3f())  // Initialize as identity
    , m_velocity(Eigen::Vector3f::Zero())  // Initialize velocity as zero
    , m_accel_bias(Eigen::Vector3f::Zero())  // Initialize accel bias as zero
    , m_gyro_bias(Eigen::Vector3f::Zero())   // Initialize gyro
    , m_dt_from_last_keyframe(0.0)          // Initialize dt as zero
    , m_T_relative_from_ref(Eigen::Matrix4f::Identity())
    , m_depth_map(depth_map.clone())
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
    
    // Process depth map
    process_depth_map(depth_map);
    
    // Compute undistorted image boundaries
    undistort_corner_points();

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

void Frame::set_Twc(const Eigen::Matrix4f& T_wc) {
    // T_wc is camera pose in world frame
    // T_wb = T_wc * T_cb = T_wc * T_bc^-1
    // For now, if T_CB is identity, T_wb = T_wc
    std::lock_guard<std::mutex> lock(m_pose_mutex);
    
    // Apply camera-to-body transformation if available
    Eigen::Matrix4f T_cb = m_T_CB.cast<float>();

    // debug

    std::cout<<T_cb<<"\n";

    Eigen::Matrix4f T_wb = T_wc * T_cb;
    
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
        // spdlog::error("[FRAME] No reference keyframe available for non-keyframe pose retrieval!");
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
            
            if (map_point && !map_point->is_bad()) {
                // Only draw features with valid MapPoints in green
                cv::Scalar point_color = cv::Scalar(0, 255, 0); // Green in BGR format
                cv::circle(display_image, pt, 4, point_color, 2);
            }
            // Features without MapPoints are not drawn (clean visualization)
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

bool Frame::is_in_border(const cv::Point2f& point, int border_size) const {
    // Undistort the point first to get its undistorted coordinates
    cv::Point2f undistorted_point = m_camera->undistort_point(point);

    // Check against undistorted boundaries
    return (m_undist_x_min + border_size <= undistorted_point.x && 
            undistorted_point.x <= m_undist_x_max - border_size && 
            m_undist_y_min + border_size <= undistorted_point.y && 
            undistorted_point.y <= m_undist_y_max - border_size);
}

void Frame::undistort_corner_points() {
    if (m_left_image.empty()) {
        // If no image is available, use default boundaries
        m_undist_x_min = 0.0;
        m_undist_x_max = 640.0;
        m_undist_y_min = 0.0;
        m_undist_y_max = 480.0;
        return;
    }
    
    int width = m_left_image.cols;
    int height = m_left_image.rows;
    
    const Config& config = Config::getInstance();
    CameraModel camera_model = config.get_camera_model();
    int border_size = config.m_border_size;
    
    if (camera_model == CameraModel::PINHOLE) {
        // For pinhole camera, undistorted boundaries are same as image boundaries (minus border)
        m_undist_x_min = static_cast<double>(border_size);
        m_undist_x_max = static_cast<double>(width - border_size);
        m_undist_y_min = static_cast<double>(border_size);
        m_undist_y_max = static_cast<double>(height - border_size);
    } else {
        // For fisheye camera, undistort the 4 corner points (with border considered)
        std::vector<cv::Point2f> corner_points = {
            cv::Point2f(static_cast<float>(border_size), static_cast<float>(border_size)),                                              // Top-left
            cv::Point2f(static_cast<float>(width - border_size), static_cast<float>(border_size)),                                      // Top-right
            cv::Point2f(static_cast<float>(border_size), static_cast<float>(height - border_size)),                                     // Bottom-left
            cv::Point2f(static_cast<float>(width - border_size), static_cast<float>(height - border_size))                              // Bottom-right
        };
        
        std::vector<cv::Point2f> undistorted_corners(4);
        for (size_t i = 0; i < corner_points.size(); ++i) {
            undistorted_corners[i] = m_camera->undistort_point(corner_points[i]);
        }
        
        // Find min/max from undistorted corners
        m_undist_x_min = undistorted_corners[0].x;
        m_undist_x_max = undistorted_corners[0].x;
        m_undist_y_min = undistorted_corners[0].y;
        m_undist_y_max = undistorted_corners[0].y;
        
        for (const auto& corner : undistorted_corners) {
            if (corner.x < m_undist_x_min) m_undist_x_min = corner.x;
            if (corner.x > m_undist_x_max) m_undist_x_max = corner.x;
            if (corner.y < m_undist_y_min) m_undist_y_min = corner.y;
            if (corner.y > m_undist_y_max) m_undist_y_max = corner.y;
        }
    }
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
    
    // Process all features - undistort using Camera class
    for (auto& feature : m_features) {
        if (feature->is_valid()) {
            // Get original pixel coordinate
            cv::Point2f pixel_pt = feature->get_pixel_coord();
            
            // Undistort left feature using left camera
            if (m_camera) {
                cv::Point2f undistorted_pixel = m_camera->undistort_point(pixel_pt);
                feature->set_undistorted_coord(undistorted_pixel);
                
                // Compute normalized camera coordinates
                Eigen::Vector2f normalized = m_camera->compute_normalized(undistorted_pixel);
                feature->set_normalized_coord(normalized);
            }
            
            // For stereo matches, undistort right coordinate using right camera
            if (feature->has_stereo_match() && m_right_camera) {
                cv::Point2f right_pixel = feature->get_right_coord();
                
                // Check if stereo match is valid
                if (right_pixel.x >= 0 && right_pixel.y >= 0) {
                    // Undistort right pixel using right camera
                    cv::Point2f right_undistorted = m_right_camera->undistort_point(right_pixel);
                    
                    // Compute normalized coordinates using right camera parameters
                    Eigen::Vector2f right_normalized = m_right_camera->compute_normalized(right_undistorted);
                    
                    // Store right normalized coordinate
                    feature->set_undistorted_stereo_match(right_normalized, -1.0f);
                } else {
                    // Invalid stereo match
                    feature->set_undistorted_stereo_match(Eigen::Vector2f(-1,-1), -1.0f);
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
            Eigen::Vector2f right_norm = feature->get_right_normalized_coord();
            if (right_norm.x() >= 0 && right_norm.y() >= 0) {
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
            double max_reproj_error = sqrt(1.0) / std::min(K_left_eigen(0,0), K_left_eigen(1,1));
            
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

// Camera parameter getters (return from member variables)
float Frame::get_fx() const {
    return static_cast<float>(m_fx);
}

float Frame::get_fy() const {
    return static_cast<float>(m_fy);
}

float Frame::get_cx() const {
    return static_cast<float>(m_cx);
}

float Frame::get_cy() const {
    return static_cast<float>(m_cy);
}

void Frame::set_distortion_coeffs(const std::vector<double>& distortion_coeffs) {
    m_distortion_coeffs = distortion_coeffs;
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

void Frame::compute_depth() {
    // RGBD depth computation - extract depth directly from depth map
    // No triangulation needed, just read from depth map at feature locations
    
    // Note: For RGBD, features are already processed by compute_rgbd_3d()
    // which handles undistortion internally, so we don't need to call undistort_features() again
    
    // Update depth array from depth map

    // Then undistort features
    undistort_features();

    m_depths.assign(m_features.size(), -1.0);
    
    for (size_t i = 0; i < m_features.size(); ++i) {
        auto feature = m_features[i];
        if (feature && feature->is_valid()) {
            // Use pixel coordinates directly (compute_rgbd_3d already handles distortion)
            cv::Point2f pixel = feature->get_undistorted_coord();
            
            // Get depth value directly from depth map (no interpolation)
            int u = static_cast<int>(std::round(pixel.x));
            int v = static_cast<int>(std::round(pixel.y));
            
            float depth = 0.0f;
            if (u >= 0 && u < m_depth_map.cols && v >= 0 && v < m_depth_map.rows) {
                depth = m_depth_map.at<float>(v, u);
            }
            
            if (depth > 0.0f) {
                // Compute 3D point in camera frame using pinhole projection
                // X = (u - cx) / fx * Z
                // Y = (v - cy) / fy * Z
                // Z = depth
                float x_cam = (pixel.x - static_cast<float>(get_cx())) / static_cast<float>(get_fx()) * depth;
                float y_cam = (pixel.y - static_cast<float>(get_cy())) / static_cast<float>(get_fy()) * depth;
                float z_cam = depth;
                
                Eigen::Vector3f point_cam(x_cam, y_cam, z_cam);
                
                m_depths[i] = static_cast<double>(depth);
                feature->set_3d_point(point_cam);
            }
        }
    }
}

double Frame::get_depth(int feature_index) const {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_depths.size())) {
        return m_depths[feature_index];
    }
    return -1.0;
}

void Frame::set_depth(int feature_index, double depth) {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_depths.size())) {
        m_depths[feature_index] = depth;
    }
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

// ⭐ ========================================================================
// RGBD-specific methods
// ========================================================================

void Frame::process_depth_map(const cv::Mat& raw_depth) {
    if (raw_depth.empty()) {
        spdlog::warn("Frame {}: Empty depth map received!", m_frame_id);
        return;
    }
    
    const Config& config = Config::getInstance();
    float depth_scale = config.m_rgbd_depth_scale;  // ⭐ Read from config (default: 1000.0)
    float min_depth = config.m_min_depth;
    float max_depth = config.m_max_depth;
    
    // Convert depth format if needed
    if (raw_depth.type() == CV_16UC1) {
        // Convert uint16 to float: depth_in_meters = raw_value / depth_scale
        raw_depth.convertTo(m_depth_map, CV_32FC1, 1.0 / depth_scale);
    } else if (raw_depth.type() == CV_32FC1) {
        // Already float (assumed in meters), just clone
        m_depth_map = raw_depth.clone();
    } else {
        spdlog::warn("Frame {}: Unsupported depth map type: {}", m_frame_id, raw_depth.type());
        return;
    }
    
    // Create validity mask
    cv::Mat valid_mask = (m_depth_map > min_depth) & (m_depth_map < max_depth);
    
    // Set invalid depths to 0
    m_depth_map.setTo(0.0f, ~valid_mask);
    
    // ⭐ No filtering, no hole filling - use raw depth as is
}

void Frame::preprocess_depth_map() {
    // ⭐ Disabled - using raw depth only
    // No bilateral filtering, no hole filling
    return;
}

float Frame::get_depth_at(float u, float v) const {
    if (m_depth_map.empty()) return 0.0f;
    
    const Config& config = Config::getInstance();
    float min_depth = config.m_min_depth;
    float max_depth = config.m_max_depth;
    
    int x = static_cast<int>(std::round(u));
    int y = static_cast<int>(std::round(v));
    
    // Boundary check
    if (x < 0 || x >= m_depth_map.cols || y < 0 || y >= m_depth_map.rows) {
        return 0.0f;
    }
    
    float depth = m_depth_map.at<float>(y, x);
    
    // Validate depth
    if (depth <= min_depth || depth >= max_depth) {
        return 0.0f;
    }
    
    return depth;
}

float Frame::get_interpolated_depth(float u, float v) const {
    if (m_depth_map.empty()) return 0.0f;
    
    const Config& config = Config::getInstance();
    float min_depth = config.m_min_depth;
    float max_depth = config.m_max_depth;
    
    // Floor coordinates
    int x0 = static_cast<int>(std::floor(u));
    int y0 = static_cast<int>(std::floor(v));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // Boundary check
    if (x0 < 0 || x1 >= m_depth_map.cols || y0 < 0 || y1 >= m_depth_map.rows) {
        return 0.0f;
    }
    
    // Get four corner depths
    float d00 = m_depth_map.at<float>(y0, x0);
    float d01 = m_depth_map.at<float>(y1, x0);
    float d10 = m_depth_map.at<float>(y0, x1);
    float d11 = m_depth_map.at<float>(y1, x1);
    
    // Check if any corner is invalid
    if (d00 <= 0 || d01 <= 0 || d10 <= 0 || d11 <= 0) {
        // Fallback to nearest neighbor
        return get_depth_at(u, v);
    }
    
    // Bilinear interpolation
    float wx = u - x0;
    float wy = v - y0;
    
    float depth = (1 - wx) * (1 - wy) * d00 +
                  wx * (1 - wy) * d10 +
                  (1 - wx) * wy * d01 +
                  wx * wy * d11;
    
    // Validate interpolated depth
    if (depth <= min_depth || depth >= max_depth) {
        return 0.0f;
    }
    
    return depth;
}

bool Frame::has_valid_depth_at(float u, float v) const {
    float depth = get_depth_at(u, v);
    const Config& config = Config::getInstance();
    return depth > config.m_min_depth && depth < config.m_max_depth;
}

float Frame::compute_depth_uncertainty(float depth) const {
    const Config& config = Config::getInstance();
    
    // Quadratic error model: σ² = a*d² + b*d + c
    float a = config.m_rgbd_uncertainty_a;
    float b = config.m_rgbd_uncertainty_b;
    float c = config.m_rgbd_uncertainty_c;
    
    float variance = a * depth * depth + b * depth + c;
    return std::sqrt(variance);
}

Frame::DepthStats Frame::get_depth_statistics() const {
    DepthStats stats;
    
    if (m_depth_map.empty()) {
        stats.min_valid_depth = 0.0f;
        stats.max_valid_depth = 0.0f;
        stats.mean_depth = 0.0f;
        stats.valid_pixels = 0;
        stats.total_pixels = 0;
        stats.valid_ratio = 0.0f;
        return stats;
    }
    
    const Config& config = Config::getInstance();
    float min_depth = config.m_min_depth;
    float max_depth = config.m_max_depth;
    
    stats.total_pixels = m_depth_map.rows * m_depth_map.cols;
    stats.valid_pixels = 0;
    stats.min_valid_depth = std::numeric_limits<float>::max();
    stats.max_valid_depth = 0.0f;
    double sum_depth = 0.0;
    
    for (int y = 0; y < m_depth_map.rows; ++y) {
        for (int x = 0; x < m_depth_map.cols; ++x) {
            float depth = m_depth_map.at<float>(y, x);
            if (depth > min_depth && depth < max_depth) {
                stats.valid_pixels++;
                sum_depth += depth;
                stats.min_valid_depth = std::min(stats.min_valid_depth, depth);
                stats.max_valid_depth = std::max(stats.max_valid_depth, depth);
            }
        }
    }
    
    stats.mean_depth = stats.valid_pixels > 0 ? 
                      static_cast<float>(sum_depth / stats.valid_pixels) : 0.0f;
    stats.valid_ratio = static_cast<float>(stats.valid_pixels) / stats.total_pixels;
    
    return stats;
}

// ⭐ RGBD Dense Point Cloud Generation
std::vector<Eigen::Vector3f> Frame::generate_dense_point_cloud(
    int stride,
    float min_depth,
    float max_depth
) const {
    std::vector<Eigen::Vector3f> points;
    
    // Check if this is an RGBD frame with depth map
    if (m_frame_type != FrameType::RGBD || m_depth_map.empty()) {
        return points;
    }
    
    // Reserve approximate space (accounting for stride and invalid depths)
    int approx_points = (m_depth_map.rows / stride) * (m_depth_map.cols / stride);
    points.reserve(approx_points / 2);  // Assume ~50% valid depths
    
    // Get camera pose (world to camera transform)
    Eigen::Matrix4f Twc = get_Twc();
    Eigen::Matrix3f R_wc = Twc.block<3, 3>(0, 0);
    Eigen::Vector3f t_wc = Twc.block<3, 1>(0, 3);
    
    // Iterate through depth map with stride
    for (int v = 0; v < m_depth_map.rows; v += stride) {
        for (int u = 0; u < m_depth_map.cols; u += stride) {
            float depth = m_depth_map.at<float>(v, u);
            
            // Check depth validity
            if (depth <= min_depth || depth >= max_depth) {
                continue;
            }
            
            // Back-project pixel to 3D camera coordinates
            float x_cam = (u - get_cx()) * depth / get_fx();
            float y_cam = (v - get_cy()) * depth / get_fy();
            float z_cam = depth;
            
            Eigen::Vector3f P_cam(x_cam, y_cam, z_cam);
            
            // Transform to world coordinates
            Eigen::Vector3f P_world = R_wc * P_cam + t_wc;
            
            points.push_back(P_world);
        }
    }
    
    return points;
}

std::vector<Frame::ColoredPoint> Frame::generate_colored_point_cloud(
    int stride,
    float min_depth,
    float max_depth,
    int color_mode
) const {
    std::vector<ColoredPoint> colored_points;
    
    // Check if this is an RGBD frame with depth map
    if (m_frame_type != FrameType::RGBD || m_depth_map.empty()) {
        return colored_points;
    }
    
    // Reserve approximate space
    int approx_points = (m_depth_map.rows / stride) * (m_depth_map.cols / stride);
    colored_points.reserve(approx_points / 2);
    
    // Get camera pose
    Eigen::Matrix4f Twc = get_Twc();
    Eigen::Matrix3f R_wc = Twc.block<3, 3>(0, 0);
    Eigen::Vector3f t_wc = Twc.block<3, 1>(0, 3);
    
    // Check if RGB image is available for color mode 1 (use m_rgb_image, not m_left_image!)
    bool has_rgb = !m_rgb_image.empty() && m_rgb_image.channels() >= 3;
    
    // Iterate through depth map with stride
    for (int v = 0; v < m_depth_map.rows; v += stride) {
        for (int u = 0; u < m_depth_map.cols; u += stride) {
            float depth = m_depth_map.at<float>(v, u);
            
            // Check depth validity
            if (depth <= min_depth || depth >= max_depth) {
                continue;
            }
            
            // Back-project to 3D camera coordinates
            float x_cam = (u - get_cx()) * depth / get_fx();
            float y_cam = (v - get_cy()) * depth / get_fy();
            float z_cam = depth;
            
            Eigen::Vector3f P_cam(x_cam, y_cam, z_cam);
            
            // Transform to world coordinates
            Eigen::Vector3f P_world = R_wc * P_cam + t_wc;
            
            // Determine color based on mode
            Eigen::Vector3f color;
            
            switch (color_mode) {
                case 0: // Mono (cyan)
                    color = Eigen::Vector3f(0.0f, 1.0f, 1.0f);
                    break;
                    
                case 1: // RGB from image
                    if (has_rgb) {
                        cv::Vec3b bgr = m_rgb_image.at<cv::Vec3b>(v, u);
                        // Convert BGR to RGB and normalize to [0, 1]
                        color = Eigen::Vector3f(
                            bgr[2] / 255.0f,  // R
                            bgr[1] / 255.0f,  // G
                            bgr[0] / 255.0f   // B
                        );
                    } else {
                        // Fallback to cyan if no RGB available
                        color = Eigen::Vector3f(0.0f, 1.0f, 1.0f);
                    }
                    break;
                    
                case 2: // Depth heatmap (near=red, far=blue)
                {
                    float normalized_depth = (depth - min_depth) / (max_depth - min_depth);
                    normalized_depth = std::max(0.0f, std::min(1.0f, normalized_depth));
                    
                    // Heatmap: Blue (0) -> Cyan -> Green -> Yellow -> Red (1)
                    if (normalized_depth < 0.25f) {
                        float t = normalized_depth / 0.25f;
                        color = Eigen::Vector3f(0.0f, t, 1.0f);  // Blue to Cyan
                    } else if (normalized_depth < 0.5f) {
                        float t = (normalized_depth - 0.25f) / 0.25f;
                        color = Eigen::Vector3f(0.0f, 1.0f, 1.0f - t);  // Cyan to Green
                    } else if (normalized_depth < 0.75f) {
                        float t = (normalized_depth - 0.5f) / 0.25f;
                        color = Eigen::Vector3f(t, 1.0f, 0.0f);  // Green to Yellow
                    } else {
                        float t = (normalized_depth - 0.75f) / 0.25f;
                        color = Eigen::Vector3f(1.0f, 1.0f - t, 0.0f);  // Yellow to Red
                    }
                    break;
                }
                
                default:
                    color = Eigen::Vector3f(0.0f, 1.0f, 1.0f);  // Default cyan
                    break;
            }
            
            colored_points.push_back({P_world, color, depth});
        }
    }
    
    return colored_points;
}

} // namespace lightweight_vio
