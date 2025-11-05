/**
 * @file      Feature.cpp
 * @brief     Implements the Feature class.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-11
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "database/Feature.h"
#include <cmath>

namespace lightweight_vio {

Feature::Feature(int feature_id, const cv::Point2f& pixel_coord)
    : m_feature_id(feature_id)
    , m_tracked_feature_id(-1)  // No tracked feature by default
    , m_pixel_coord(pixel_coord)
    , m_undistorted_coord(cv::Point2f(-1, -1))  // Invalid until computed
    , m_normalized_coord(Eigen::Vector2f::Zero())
    , m_3d_point(Eigen::Vector3f::Zero())
    , m_velocity(Eigen::Vector2f::Zero())
    , m_track_count(1)
    , m_num_observations_accumulated(1)  // Start with 1 since we're creating this observation
    , m_depth(-1.0f)  // Invalid depth initially
    , m_reprojection_error(-1.0f)  // Invalid reprojection error initially
    , m_is_valid(true)
    , m_has_3d_point(false)
    , m_right_coord(cv::Point2f(-1, -1))  // Invalid stereo coordinate initially
    , m_right_normalized_coord(Eigen::Vector2f(-1, -1))  // Invalid normalized stereo coordinate initially
    , m_disparity(-1.0f)  // Invalid disparity initially
    , m_undistorted_disparity(-1.0f)  // Invalid undistorted disparity initially
    , m_has_stereo_match(false)
{
}

} // namespace lightweight_vio
