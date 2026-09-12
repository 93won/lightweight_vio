/**
 * @file      Factors.cpp
 * @brief     Implements Ceres cost functions (factors) for VIO optimization.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-28
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "optimization/Factors.h"
#include "processing/IMUHandler.h"  // For IMUPreintegration
#include <limits>
#include <spdlog/spdlog.h>
namespace lightweight_vio {
namespace factor {

// PnPFactor implementation
PnPFactor::PnPFactor(const Eigen::Vector2d& observation,
                            const Eigen::Vector3d& world_point,
                            const CameraParameters& camera_params,
                            const Eigen::Matrix4d& Tcb,
                            const Eigen::Matrix2d& information)
    : m_observation(observation), m_world_point(world_point), 
      m_camera_params(camera_params), m_Tcb(Tcb), m_information(information), m_is_outlier(false) {}

bool PnPFactor::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    // If marked as outlier, set residuals to zero and jacobians to zero
    if (m_is_outlier) {
        residuals[0] = 0.0;
        residuals[1] = 0.0;
        
        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac(jacobians[0]);
            jac.setZero();
        }
        return true;
    }

    // Extract SE3 parameters from tangent space (Ceres order: tx,ty,tz,rx,ry,rz)
    Eigen::Map<const Eigen::Vector6d> se3_tangent(parameters[0]);
    
    // Convert tangent space to SE3 using Sophus exp (consistent with parameterization)
    Sophus::SE3d T_wb = Sophus::SE3d::exp(se3_tangent);
    
    // Convert T_wb to T_cw transformation
    Eigen::Matrix3d R_wb = T_wb.rotationMatrix();
    Eigen::Vector3d t_wb = T_wb.translation();
    Eigen::Matrix3d R_bw = R_wb.transpose();
    Eigen::Vector3d t_bw = -R_bw * t_wb;
    
    // T_cw = T_cb * T_bw
    Eigen::Matrix3d R_cw = m_Tcb.block<3, 3>(0, 0) * R_bw;
    Eigen::Vector3d t_cw = m_Tcb.block<3, 3>(0, 0) * t_bw + m_Tcb.block<3, 1>(0, 3);
    
    // Transform world point to camera coordinates: Pc = R_cw * Pw + t_cw
    Eigen::Vector3d point_camera = R_cw * m_world_point + t_cw;
    
    double x = point_camera.x();
    double y = point_camera.y();
    double z = point_camera.z();
    
    // Check for valid depth
    if (z <= 1e-6) {
        residuals[0] = 0.0;
        residuals[1] = 0.0;
        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac(jacobians[0]);
            jac.setZero();
        }
        return true;
    }
    
    double z_inv = 1.0 / z;
    
    // Project to image plane
    double u = m_camera_params.fx * x * z_inv + m_camera_params.cx;
    double v = m_camera_params.fy * y * z_inv + m_camera_params.cy;
    
    // Compute residuals: observation - projection
    Eigen::Vector2d residual_vec;
    residual_vec << m_observation.x() - u, m_observation.y() - v;
    
    // Apply information matrix weighting to residuals: r_weighted = sqrt(Info) * r
    Eigen::LLT<Eigen::Matrix2d> llt(m_information);
    if (llt.info() == Eigen::Success) {
        // Use Cholesky decomposition: Information = L * L^T
        // Weighted residual = L * residual
        Eigen::Vector2d weighted_residual = llt.matrixL() * residual_vec;
        residuals[0] = weighted_residual[0];
        residuals[1] = weighted_residual[1];
    } else {
        // Fallback to unweighted if Cholesky fails
        residuals[0] = residual_vec[0];
        residuals[1] = residual_vec[1];
    }


    
    // Compute analytical jacobians if requested
    if (jacobians && jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac(jacobians[0]);
        
        double z_inv_sq = z_inv * z_inv;
        
        // Jacobian calculation
        // Camera extrinsic rotation matrix
        Eigen::Matrix3d Rcb = m_Tcb.block<3, 3>(0, 0);
        
        // Body frame coordinates: Pb = Rbw * Pw + tbw
        Eigen::Vector3d Pb = R_bw * m_world_point + t_bw;
        
        // Jacobian of projection error w.r.t. camera coordinates: ∂(error)/∂(Pc)
        Eigen::Matrix<double, 2, 3> J_error_wrt_Pc;
        J_error_wrt_Pc(0, 0) = -m_camera_params.fx * z_inv;
        J_error_wrt_Pc(0, 1) = 0.0;
        J_error_wrt_Pc(0, 2) = x * m_camera_params.fx * z_inv_sq;
        J_error_wrt_Pc(1, 0) = 0.0;
        J_error_wrt_Pc(1, 1) = -m_camera_params.fy * z_inv;
        J_error_wrt_Pc(1, 2) = y * m_camera_params.fy * z_inv_sq;
        
        // Chain rule: ∂(error)/∂(twist) = ∂(error)/∂(Pc) * ∂(Pc)/∂(twist)
        
        // Translation part: ∂(Pc)/∂(translation) = -Rcb
        Eigen::Matrix<double, 2, 3> J_translation = J_error_wrt_Pc * (-Rcb);
        
        // Rotation part: ∂(Pc)/∂(rotation) = Rcb * [Pb]× for right perturbation
        Eigen::Matrix3d hatPb = Sophus::SO3d::hat(Pb);
        Eigen::Matrix<double, 2, 3> J_rotation = J_error_wrt_Pc * Rcb * hatPb;
        
        // Combine Jacobian components [translation | rotation] for Ceres order
        Eigen::Matrix<double, 2, 6> unweighted_jac;
        unweighted_jac << J_translation, J_rotation;
        
        // Apply information matrix weighting to Jacobian: J_weighted = sqrt(Info) * J
        Eigen::LLT<Eigen::Matrix2d> llt(m_information);
        if (llt.info() == Eigen::Success) {
            jac = llt.matrixL() * unweighted_jac;
        } else {
            jac = unweighted_jac;
        }
    }
    
    return true;
}

double PnPFactor::compute_chi_square(double const* const* parameters) const {
    // Extract SE3 parameters from tangent space (Ceres order: tx,ty,tz,rx,ry,rz)
    Eigen::Map<const Eigen::Vector6d> se3_tangent(parameters[0]);
    
    // Convert tangent space to SE3 using Sophus exp (consistent with parameterization)
    Sophus::SE3d T_wb = Sophus::SE3d::exp(se3_tangent);
    
    // Convert T_wb to T_cw transformation
    Eigen::Matrix3d R_wb = T_wb.rotationMatrix();
    Eigen::Vector3d t_wb = T_wb.translation();
    Eigen::Matrix3d R_bw = R_wb.transpose();
    Eigen::Vector3d t_bw = -R_bw * t_wb;
    
    // T_cw = T_cb * T_bw
    Eigen::Matrix3d R_cw = m_Tcb.block<3, 3>(0, 0) * R_bw;
    Eigen::Vector3d t_cw = m_Tcb.block<3, 3>(0, 0) * t_bw + m_Tcb.block<3, 1>(0, 3);
    
    // Transform world point to camera coordinates: Pc = R_cw * Pw + t_cw
    Eigen::Vector3d point_camera = R_cw * m_world_point + t_cw;
    
    double x = point_camera.x();
    double y = point_camera.y();
    double z = point_camera.z();
    
    // Check for valid depth
    if (z <= 1e-6) {
        
        return std::numeric_limits<double>::max(); // Invalid, return large chi-square
    }
    
    double z_inv = 1.0 / z;
    
    // Project to image plane
    double u = m_camera_params.fx * x * z_inv + m_camera_params.cx;
    double v = m_camera_params.fy * y * z_inv + m_camera_params.cy;
    
    // Compute residuals: observation - projection
    Eigen::Vector2d residual_vec;
    residual_vec << m_observation.x() - u, m_observation.y() - v;
    
    // Chi-square error with information matrix: r^T * Information * r
    double chi2_error = residual_vec.transpose() * m_information * residual_vec;
    
    return chi2_error;
}

// BAFactor implementation
BAFactor::BAFactor(const Eigen::Vector2d& observation,
                   const CameraParameters& camera_params,
                   const Eigen::Matrix4d& Tcb,
                   const Eigen::Matrix2d& information)
    : m_observation(observation)
    , m_camera_params(camera_params)
    , m_Tcb(Tcb)
    , m_information(information)
    , m_is_outlier(false) {
}

bool BAFactor::Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    
    if (m_is_outlier) {
        // If marked as outlier, set zero residual and jacobians
        residuals[0] = 0.0;
        residuals[1] = 0.0;
        
        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac_pose(jacobians[0]);
                jac_pose.setZero();
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_point(jacobians[1]);
                jac_point.setZero();
            }
        }
        return true;
    }
    
    try {
        // Extract pose parameters (SE3 tangent space)
        Eigen::Map<const Eigen::Vector6d> pose_tangent(parameters[0]);
        
        // Extract 3D point in world coordinates
        Eigen::Map<const Eigen::Vector3d> world_point(parameters[1]);
        
        // Convert tangent space to SE3 (Twb: body to world)
        Sophus::SE3d Twb = Sophus::SE3d::exp(pose_tangent);
        
        // Get rotation and translation matrices
        Eigen::Matrix3d Rwb = Twb.rotationMatrix();
        Eigen::Vector3d twb = Twb.translation();
        Eigen::Matrix3d Rbw = Rwb.transpose();
        Eigen::Vector3d tbw = -Rbw * twb;
        
        // Compute Tcw (camera to world transformation)
        // Tcw = Tcb * Tbw = Tcb * Twb^(-1)
        Eigen::Matrix3d Rcw = m_Tcb.block<3, 3>(0, 0) * Rbw;
        Eigen::Vector3d tcw = m_Tcb.block<3, 3>(0, 0) * tbw + m_Tcb.block<3, 1>(0, 3);
        
        // Transform world point to camera coordinates
        Eigen::Vector3d camera_point = Rcw * world_point + tcw;
        
        double x = camera_point[0];
        double y = camera_point[1];
        double z = camera_point[2];
        double invz = 1.0 / (z + 1e-9);

        // Check for valid depth
        if (invz < 1e-3 || invz > 1e2)
        {
            residuals[0] = 0.0;
            residuals[1] = 0.0;

            if (jacobians)
            {
                if (jacobians[0])
                {
                    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac_pose(jacobians[0]);
                    jac_pose.setZero();
                }
                if (jacobians[1])
                {
                    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_point(jacobians[1]);
                    jac_point.setZero();
                }
            }
            return true;
        }

        // Project to pixel coordinates
        double u = m_camera_params.fx * x * invz + m_camera_params.cx;
        double v = m_camera_params.fy * y * invz + m_camera_params.cy;
        
        // Compute residual: observed - projected
        Eigen::Vector2d projected(u, v);
        Eigen::Vector2d error = m_observation - projected;
        
        // Apply information matrix weighting using Cholesky decomposition: r_weighted = sqrt(Info) * r
        Eigen::LLT<Eigen::Matrix2d> llt(m_information);
        if (llt.info() == Eigen::Success) {
            // Use Cholesky decomposition: Information = L * L^T
            // Weighted residual = L * residual
            Eigen::Vector2d weighted_error = llt.matrixL() * error;
            residuals[0] = weighted_error[0];
            residuals[1] = weighted_error[1];
        } else {
            // Fallback to unweighted if Cholesky fails
            residuals[0] = error[0];
            residuals[1] = error[1];
        }
        
        // Compute Jacobians if requested
        if (jacobians) {
            // Camera projection Jacobian w.r.t camera point
            Eigen::Matrix<double, 2, 3> J_proj_camera;
            J_proj_camera(0, 0) = -m_camera_params.fx * invz;
            J_proj_camera(0, 1) = 0.0;
            J_proj_camera(0, 2) = x * m_camera_params.fx * invz * invz;
            J_proj_camera(1, 0) = 0.0;
            J_proj_camera(1, 1) = -m_camera_params.fy * invz;
            J_proj_camera(1, 2) = y * m_camera_params.fy * invz * invz;
            
            // Apply information matrix weighting to projection jacobian using Cholesky
            Eigen::LLT<Eigen::Matrix2d> llt(m_information);
            Eigen::Matrix<double, 2, 3> weighted_J_proj;
            if (llt.info() == Eigen::Success) {
                weighted_J_proj = llt.matrixL() * J_proj_camera;
            } else {
                weighted_J_proj = J_proj_camera;
            }
            
            // Jacobian w.r.t pose (SE3 tangent space)
            if (jacobians[0]) {
                // Body point in body frame: Pb = Rbw * (Pw - twb)
                Eigen::Vector3d body_point = Rbw * (world_point - twb);
                
                // Jacobian of camera point w.r.t body translation
                Eigen::Matrix<double, 3, 3> J_camera_trans = -m_Tcb.block<3, 3>(0, 0);
                
                // Jacobian of camera point w.r.t body rotation  
                Eigen::Matrix<double, 3, 3> J_camera_rot = m_Tcb.block<3, 3>(0, 0) * Sophus::SO3d::hat(body_point);
                
                // Combine translation and rotation jacobians [3x6]
                // Ceres order: [translation(0-2) | rotation(3-5)]
                Eigen::Matrix<double, 3, 6> J_camera_pose;
                J_camera_pose.block<3, 3>(0, 0) = J_camera_trans; // w.r.t translation
                J_camera_pose.block<3, 3>(0, 3) = J_camera_rot;   // w.r.t rotation
                
                // Chain rule: J_residual_pose = J_proj_camera * J_camera_pose
                Eigen::Matrix<double, 2, 6> J_pose = weighted_J_proj * J_camera_pose;
                
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac_pose(jacobians[0]);
                jac_pose = J_pose;
            }
            
            // Jacobian w.r.t 3D point
            if (jacobians[1]) {
                // Jacobian of camera point w.r.t world point
                Eigen::Matrix<double, 3, 3> J_camera_point = m_Tcb.block<3, 3>(0, 0) * Rbw;
                
                // Chain rule: J_residual_point = J_proj_camera * J_camera_point  
                Eigen::Matrix<double, 2, 3> J_point = weighted_J_proj * J_camera_point;
                
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_point(jacobians[1]);
                jac_point = J_point;
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

double BAFactor::compute_chi_square(double const* const* parameters) const {
    if (m_is_outlier) {
        return 0.0;
    }
    
    // Compute residual (which is already weighted by information matrix in Evaluate())
    double residuals[2];
    Evaluate(parameters, residuals, nullptr);
    
    // Chi-square error: weighted_residual^T * weighted_residual
    // Since Evaluate() returns Information * error, this gives us error^T * Information^T * Information * error
    // For symmetric positive definite Information matrix: Information^T = Information
    // So this becomes: error^T * Information^2 * error, which is incorrect for chi-square test
    
    // We need to compute the unweighted error and then apply information matrix properly
    // Let's recompute the unweighted error for proper chi-square calculation
    
    // Extract SE3 pose from parameters
    Eigen::Map<const Eigen::Vector6d> se3_tangent(parameters[0]);
    Sophus::SE3d Twb = Sophus::SE3d::exp(se3_tangent);
    
    // Extract 3D point
    Eigen::Map<const Eigen::Vector3d> world_point(parameters[1]);
    
    // Transform world point to camera coordinates
    Eigen::Matrix3d Rwb = Twb.rotationMatrix();
    Eigen::Vector3d twb = Twb.translation();
    Eigen::Matrix3d Rbw = Rwb.transpose();
    
    // Body coordinates: Pb = Rbw * (Pw - twb)
    Eigen::Vector3d body_point = Rbw * (world_point - twb);
    
    // Camera coordinates: Pc = Tcb * [Pb; 1]
    Eigen::Vector4d body_point_h(body_point.x(), body_point.y(), body_point.z(), 1.0);
    Eigen::Vector4d camera_point_h = m_Tcb * body_point_h;
    Eigen::Vector3d camera_point = camera_point_h.head<3>();
    
    // Check if point is in front of camera
    if (camera_point.z() <= 0.0) {
        
        return 1000.0; // Large chi-square for points behind camera
    }
    
    // Project to pixel coordinates
    double invz = 1.0 / camera_point.z();
    double u = m_camera_params.fx * camera_point.x() * invz + m_camera_params.cx;
    double v = m_camera_params.fy * camera_point.y() * invz + m_camera_params.cy;
    
    // Compute unweighted error
    Eigen::Vector2d projected(u, v);
    Eigen::Vector2d error = m_observation - projected;
    
    // Proper chi-square: error^T * Information * error
    double chi_square = error.transpose() * m_information * error;
    
    return chi_square;
}

// ===============================================================================
// INERTIAL GRAVITY FACTOR IMPLEMENTATION
// ===============================================================================

InertialGravityFactor::InertialGravityFactor(std::shared_ptr<IMUPreintegration> preintegration,
                                           double gravity_magnitude)
    : m_preintegration(preintegration), m_gravity_magnitude(gravity_magnitude) {
    // Extract covariance for rotation, velocity, and position (9x9 block)
    Eigen::Matrix<double, 9, 9> covariance_9x9 = m_preintegration->covariance.block<9, 9>(0, 0).cast<double>();
    
    // Compute information matrix (covariance inverse) with numerical stability check
    Eigen::JacobiSVD<Eigen::Matrix<double, 9, 9>> svd(covariance_9x9, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    // Apply regularization for numerical stability
    const double min_singular_value = 1e-6;
    Eigen::Matrix<double, 9, 1> singular_values = svd.singularValues();
    for (int i = 0; i < 9; ++i) {
        if (singular_values(i) < min_singular_value) {
            singular_values(i) = min_singular_value;
        }
    }
    
    // Compute regularized inverse: A^(-1) = V * S^(-1) * U^T
    Eigen::Matrix<double, 9, 9> information = svd.matrixV() * singular_values.cwiseInverse().asDiagonal() * svd.matrixU().transpose();
    
    // ⭐ Scale down information matrix for better numerical conditioning
    // IMU preintegration covariance is often too optimistic (too small)
    // Scale by 1e-6 to make information ~1.0 instead of ~1e6
    information *= 1e-6;
    
    // Compute square root information matrix using Cholesky decomposition
    Eigen::LLT<Eigen::Matrix<double, 9, 9>> llt(information);
    if (llt.info() == Eigen::Success) {
        m_sqrt_information = llt.matrixL().transpose(); // Upper triangular
    } else {
        // Fallback to identity if decomposition fails
        m_sqrt_information = Eigen::Matrix<double, 9, 9>::Identity();
        spdlog::warn("[InertialGravityFactor] Cholesky decomposition failed, using identity weighting");
    }
    
}

bool InertialGravityFactor::Evaluate(double const* const* parameters,
                                    double* residuals,
                                    double** jacobians) const {
    
    // Mathematical foundation: This implementation follows the formulation in
    // "IMU Preintegration on Manifold for Efficient Visual-Inertial Maximum-a-Posteriori Estimation"
    // by Forster et al. (RSS 2015), specifically Appendix C for gravity-aware preintegration factors.
    
    // ===============================================================================
    // STEP 1: Extract parameters from optimization variables
    // ===============================================================================
    
    // parameters[0]: SE3 posei [ti, Ri] (Twb format)
    Eigen::Map<const Eigen::Vector6d> posei_tangent(parameters[0]);
    Sophus::SE3d T_wbi = Sophus::SE3d::exp(posei_tangent);
    Eigen::Matrix3d R_wbi = T_wbi.rotationMatrix();
    Eigen::Vector3d t_wbi = T_wbi.translation();
    Eigen::Matrix3d R_bwi = R_wbi.transpose();
    
    // parameters[1]: velocityi [vi]
    Eigen::Map<const Eigen::Vector3d> vi(parameters[1]);
    
    // parameters[2]: shared gyro bias [bg] - SHARED across all factors
    Eigen::Map<const Eigen::Vector3d> bg(parameters[2]);
    
    // parameters[3]: shared accel bias [ba] - SHARED across all factors  
    Eigen::Map<const Eigen::Vector3d> ba(parameters[3]);
    
    // parameters[4]: SE3 posej [tj, Rj] (Twb format)
    Eigen::Map<const Eigen::Vector6d> posej_tangent(parameters[4]);
    Sophus::SE3d T_wbj = Sophus::SE3d::exp(posej_tangent);
    Eigen::Matrix3d R_wbj = T_wbj.rotationMatrix();
    Eigen::Vector3d t_wbj = T_wbj.translation();
    
    // parameters[5]: velocityj [vj]
    Eigen::Map<const Eigen::Vector3d> vj(parameters[5]);
    
    // parameters[6]: gravity_dir: 2D gravity direction parameterization
    Eigen::Map<const Eigen::Vector2d> gravity_dir(parameters[6]);
    
    // ===============================================================================
    // STEP 2: Compute gravity vector from direction parameterization
    // ===============================================================================
    
    Eigen::Matrix3d R_wg = gravity_dir_to_rotation(gravity_dir);
    Eigen::Vector3d g_I(0, 0, -m_gravity_magnitude);
    Eigen::Vector3d g = R_wg * g_I;  // gravity in world frame
    
    // std::cout << "[InertialGravityFactor] g_I: " << g_I.transpose() << std::endl;
    // std::cout << "[InertialGravityFactor] g (world): " << g.transpose() << std::endl;
    
    // ===============================================================================
    // STEP 3: Get bias-corrected preintegration values
    // ===============================================================================
    
    double dt = m_preintegration->dt_total;
    
    // Create bias struct (assuming similar to IMU::Bias structure)
    // Note: You may need to adjust this based on your IMUPreintegration structure
    Eigen::Vector3d current_ba = ba;
    Eigen::Vector3d current_bg = bg;
    
    // Get corrected preintegration values 
    Eigen::Matrix3d delta_R = m_preintegration->delta_R.cast<double>(); // This should be GetDeltaRotation(bias)
    Eigen::Vector3d delta_V = m_preintegration->delta_V.cast<double>(); // This should be GetDeltaVelocity(bias) 
    Eigen::Vector3d delta_P = m_preintegration->delta_P.cast<double>(); // This should be GetDeltaPosition(bias)
    
    // Apply bias corrections using Jacobians
    Eigen::Vector3d delta_bg = bg - m_preintegration->gyro_bias.cast<double>();
    Eigen::Vector3d delta_ba = ba - m_preintegration->accel_bias.cast<double>();
    
    if (delta_bg.norm() > 1e-6 || delta_ba.norm() > 1e-6) {
        // Apply bias correction using precomputed Jacobians
        Eigen::Matrix3d J_Rg = m_preintegration->J_Rg.cast<double>();
        Eigen::Matrix3d J_Vg = m_preintegration->J_Vg.cast<double>();
        Eigen::Matrix3d J_Va = m_preintegration->J_Va.cast<double>();
        Eigen::Matrix3d J_Pg = m_preintegration->J_Pg.cast<double>();
        Eigen::Matrix3d J_Pa = m_preintegration->J_Pa.cast<double>();
        
        delta_R = delta_R * Sophus::SO3d::exp(J_Rg*delta_bg).matrix();
        delta_V = delta_V + J_Vg * delta_bg + J_Va * delta_ba;
        delta_P = delta_P + J_Pg * delta_bg + J_Pa * delta_ba;
    }
    
    // ===============================================================================  
    // STEP 4: Compute residuals 
    // ===============================================================================
    
    Eigen::Map<Eigen::Vector3d> er(residuals);      // rotation residual  
    Eigen::Map<Eigen::Vector3d> ev(residuals + 3);  // velocity residual
    Eigen::Map<Eigen::Vector3d> ep(residuals + 6);  // position residual
    
    // Rotation residual: er = Log(delta_R^T * Ri^T * Rj)
    er = log_SO3(delta_R.transpose() * R_bwi * R_wbj);
    
    // Velocity residual: ev = Ri^T * ((vj - vi) - g*dt) - delta_V
    ev = R_bwi * ((vj - vi) - g * dt) - delta_V;
    
    // Position residual: ep = Ri^T * ((tj - ti - vi*dt) - g*dt²/2) - delta_P  
    ep = R_bwi * ((t_wbj - t_wbi - vi * dt) - 0.5 * g * dt * dt) - delta_P;

    // ===============================================================================
    // STEP 5: Compute Jacobians 
    // ===============================================================================
    
    if (jacobians != nullptr) {
        
        // Get bias corrected values for Jacobian computation
        Eigen::Vector3d dbg = delta_bg;
        Eigen::Matrix3d eR = delta_R.transpose() * R_bwi * R_wbj;
        Eigen::Vector3d er_vec = log_SO3(eR);
        Eigen::Matrix3d Jr_inv = right_jacobian_SO3(er_vec).inverse();
        
        // Jacobian w.r.t posei [0] - parameters[0]: [6x9]
        if (jacobians[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> J_posei(jacobians[0]);
            J_posei.setZero();
            
            // translation part (columns 0-2)
            J_posei.block<3, 3>(6, 0) = -Eigen::Matrix3d::Identity();
            // rotation part (columns 3-5)
            J_posei.block<3, 3>(0, 3) = -Jr_inv * R_wbj.transpose() * R_wbi;
            J_posei.block<3, 3>(3, 3) = skew_symmetric(R_bwi * ((vj - vi) - g * dt));
            J_posei.block<3, 3>(6, 3) = skew_symmetric(R_bwi * ((t_wbj - t_wbi - vi * dt) - 0.5 * g * dt * dt));
        }
        
        // Jacobian w.r.t velocity [1] - parameters[1]: [3x9]
        if (jacobians[1] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_veli(jacobians[1]);
            J_veli.setZero();
            J_veli.block<3, 3>(3, 0) = -R_bwi;
            J_veli.block<3, 3>(6, 0) = -R_bwi * dt;
        }
        
        // Jacobian w.r.t gyro_bias [2] - parameters[2]: [3x9] 
        if (jacobians[2] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_gyro(jacobians[2]);
            J_gyro.setZero();
            
            Eigen::Matrix3d J_Rg = m_preintegration->J_Rg.cast<double>();
            Eigen::Matrix3d J_Vg = m_preintegration->J_Vg.cast<double>();
            Eigen::Matrix3d J_Pg = m_preintegration->J_Pg.cast<double>();
            
            J_gyro.block<3, 3>(0, 0) = -Jr_inv * eR.transpose() * right_jacobian_SO3(J_Rg*dbg) * J_Rg;
            J_gyro.block<3, 3>(3, 0) = -J_Vg;
            J_gyro.block<3, 3>(6, 0) = -J_Pg;
        }
        
        // Jacobian w.r.t accel_bias [3] - parameters[3]: [3x9]
        if (jacobians[3] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_accel(jacobians[3]);
            J_accel.setZero();
            
            Eigen::Matrix3d J_Va = m_preintegration->J_Va.cast<double>();
            Eigen::Matrix3d J_Pa = m_preintegration->J_Pa.cast<double>();
            
            J_accel.block<3, 3>(3, 0) = -J_Va;
            J_accel.block<3, 3>(6, 0) = -J_Pa;
        }
        
        // Jacobian w.r.t posej [4] - parameters[4]: [6x9]
        if (jacobians[4] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> J_posej(jacobians[4]);
            J_posej.setZero();
            
            // translation part (columns 0-2)
            J_posej.block<3, 3>(6, 0) = R_bwi * R_wbj;
            
            // rotation part (columns 3-5)
            J_posej.block<3, 3>(0, 3) = Jr_inv;
        }
        
        // Jacobian w.r.t velocityj [5] - parameters[5]: [3x9]
        if (jacobians[5] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_velj(jacobians[5]);
            J_velj.setZero();
            J_velj.block<3, 3>(3, 0) = R_bwi;
        }
        
        // Jacobian w.r.t gravity_dir [6] - parameters[6]: [2x9]
        if (jacobians[6] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 2, Eigen::RowMajor>> J_gravity(jacobians[6]);
            J_gravity.setZero();
            
            // Exp(w + dw) = Exp(w) Exp(Jr(w) dw), w = [theta_x, theta_y, 0].
            Eigen::Vector3d gravity_rotation(gravity_dir[0], gravity_dir[1], 0.0);
            Eigen::Matrix<double, 3, 2> gravity_parameter_basis;
            gravity_parameter_basis << 1.0, 0.0,
                                       0.0, 1.0,
                                       0.0, 0.0;
            Eigen::Matrix<double, 3, 2> dg_dtheta =
                -R_wg * skew_symmetric(g_I) *
                right_jacobian_SO3(gravity_rotation) * gravity_parameter_basis;
            
            J_gravity.block<3, 2>(3, 0) = -R_bwi * dg_dtheta * dt;
            J_gravity.block<3, 2>(6, 0) = -0.5 * R_bwi * dg_dtheta * dt * dt;
        }
    }
    
    return true;
}

Eigen::Matrix3d InertialGravityFactor::skew_symmetric(const Eigen::Vector3d& v) const {
    Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
    skew(0, 1) = -v(2);
    skew(0, 2) =  v(1);
    skew(1, 0) =  v(2);
    skew(1, 2) = -v(0);
    skew(2, 0) = -v(1);
    skew(2, 1) =  v(0);
    return skew;
}

Eigen::Matrix3d InertialGravityFactor::right_jacobian_SO3(const Eigen::Vector3d& phi) const {
    double theta = phi.norm();
    if (theta < 1e-6) {
        return Eigen::Matrix3d::Identity() - 0.5 * skew_symmetric(phi);
    }
    
    double c = cos(theta);
    double s = sin(theta);
    Eigen::Matrix3d W = skew_symmetric(phi);
    
    return Eigen::Matrix3d::Identity() - 
           W * (1.0 - c) / (theta * theta) + 
           W * W * (theta - s) / (theta * theta * theta);
}

Eigen::Matrix3d InertialGravityFactor::left_jacobian_SO3(const Eigen::Vector3d& phi) const {
    return right_jacobian_SO3(-phi).transpose();
}

Eigen::Vector3d InertialGravityFactor::log_SO3(const Eigen::Matrix3d& R) const {
    // Normalize rotation matrix using SVD to ensure orthogonality before log operation
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R_normalized = svd.matrixU() * svd.matrixV().transpose();
    
    return Sophus::SO3d(R_normalized).log();
}

Eigen::Matrix3d InertialGravityFactor::gravity_dir_to_rotation(const Eigen::Vector2d& gravity_dir) const {
    // ⭐ ORB-SLAM3 compatible implementation
    // Convert 2D perturbation to rotation matrix using SO(3) exponential map
    // Rwg = ExpSO3(theta[0], theta[1], 0.0)
    // where theta[0], theta[1] are small angles around Y and X axes respectively
    
    double theta_x = gravity_dir[0];  // rotation around Y axis
    double theta_y = gravity_dir[1];  // rotation around X axis
    
    // Use Rodrigues formula: ExpSO3([theta_x, theta_y, 0])
    Eigen::Vector3d w(theta_x, theta_y, 0.0);
    const double d2 = w.dot(w);
    const double d = std::sqrt(d2);
    
    // Skew-symmetric matrix [w]×
    Eigen::Matrix3d W;
    W << 0.0,      -w(2),    w(1),
         w(2),      0.0,    -w(0),
        -w(1),      w(0),     0.0;
    
    Eigen::Matrix3d Rwg;
    if (d < 1e-5) {
        // Small angle approximation
        Rwg = Eigen::Matrix3d::Identity() + W + 0.5*W*W;
    } else {
        // Full Rodrigues formula
        Rwg = Eigen::Matrix3d::Identity() + W*std::sin(d)/d + W*W*(1.0 - std::cos(d))/d2;
    }
    
    return Rwg;
}
} // namespace factor
} // namespace lightweight_vio
