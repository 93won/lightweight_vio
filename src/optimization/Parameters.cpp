/**
 * @file      Parameters.cpp
 * @brief     Implements parameter block management for Ceres optimization.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "optimization/Parameters.h"
#include <iostream>

namespace lightweight_vio {
namespace factor {

bool SE3GlobalParameterization::Plus(const double* x,
                                   const double* delta,
                                   double* x_plus_delta) const {
    try {
        // If parameter is fixed, don't apply any updates
        if (m_is_fixed) {
            // Copy original parameters without applying delta
            for (int i = 0; i < 6; ++i) {
                x_plus_delta[i] = x[i];
            }
            return true;
        }
        
        // Convert arrays to Eigen vectors
        Eigen::Map<const Eigen::Vector6d> current_tangent(x);
        Eigen::Map<const Eigen::Vector6d> delta_tangent(delta);
        Eigen::Map<Eigen::Vector6d> result_tangent(x_plus_delta);
        
        // Convert current tangent to SE3
        Sophus::SE3d current_se3 = TangentToSE3(current_tangent);
        
        // Apply delta as right multiplication: current * exp(delta)
        // This is appropriate for Twb (body to world) where perturbation is in body frame
        Sophus::SE3d delta_se3 = Sophus::SE3d::exp(delta_tangent);
        Sophus::SE3d result_se3 = current_se3 * delta_se3;
        
        // Convert back to tangent space
        result_tangent = SE3ToTangent(result_se3);
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool SE3GlobalParameterization::ComputeJacobian(const double* x,
                                              double* jacobian) const {
    // For small perturbations in SE3, the Jacobian can be approximated as Identity
    // This is much faster than computing the exact right Jacobian
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jac(jacobian);
    jac.setIdentity();
    return true;
}

Sophus::SE3d SE3GlobalParameterization::TangentToSE3(const Eigen::Vector6d& tangent) {
    // Ceres order: [t_x, t_y, t_z, so3_x, so3_y, so3_z]
    // Use Sophus SE3::exp for consistent parameterization with V matrix
    return Sophus::SE3d::exp(tangent);
}

Eigen::Vector6d SE3GlobalParameterization::SE3ToTangent(const Sophus::SE3d& se3) {
    // Use SE3::log() for consistency with SE3::exp() in TangentToSE3()
    // This ensures proper V matrix handling
    return se3.log();
}

bool MapPointParameterization::Plus(const double* x,
                                  const double* delta,
                                  double* x_plus_delta) const {
    try {
        // If parameter is fixed, don't apply any updates
        if (m_is_fixed) {
            // Copy original parameters without applying delta
            x_plus_delta[0] = x[0];  // x coordinate
            x_plus_delta[1] = x[1];  // y coordinate
            x_plus_delta[2] = x[2];  // z coordinate
            return true;
        }
        
        // Simple Euclidean addition for 3D points
        // x_plus_delta = x + delta
        x_plus_delta[0] = x[0] + delta[0];  // x coordinate
        x_plus_delta[1] = x[1] + delta[1];  // y coordinate
        x_plus_delta[2] = x[2] + delta[2];  // z coordinate
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool MapPointParameterization::ComputeJacobian(const double* x,
                                             double* jacobian) const {
    // For Euclidean 3D points, the Jacobian of Plus operation w.r.t delta is Identity
    // d(x + delta)/d(delta) = I (3x3 identity matrix)
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac(jacobian);
    jac.setIdentity();
    return true;
}

// ===============================================================================
// VELOCITY PARAMETERIZATION IMPLEMENTATION
// ===============================================================================

bool VelocityParameterization::Plus(const double* x,
                                   const double* delta,
                                   double* x_plus_delta) const {
    try {
        // If parameter is fixed, don't apply any updates
        if (m_is_fixed) {
            // Copy original parameters without applying delta
            for (int i = 0; i < 3; ++i) {
                x_plus_delta[i] = x[i];
            }
            return true;
        }
        
        // Simple addition for Euclidean velocity parameters
        for (int i = 0; i < 3; ++i) {
            x_plus_delta[i] = x[i] + delta[i];
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool VelocityParameterization::ComputeJacobian(const double* x,
                                              double* jacobian) const {
    // For Euclidean 3D velocity, the Jacobian of Plus operation w.r.t delta is Identity
    // d(x + delta)/d(delta) = I (3x3 identity matrix)
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac(jacobian);
    jac.setIdentity();
    return true;
}

// ===============================================================================
// BIAS PARAMETERIZATION IMPLEMENTATION
// ===============================================================================

bool BiasParameterization::Plus(const double* x,
                               const double* delta,
                               double* x_plus_delta) const {
    try {
        // If parameter is fixed, don't apply any updates
        if (m_is_fixed) {
            // Copy original parameters without applying delta
            for (int i = 0; i < 3; ++i) {
                x_plus_delta[i] = x[i];
            }
            return true;
        }
        
        // Simple addition for Euclidean bias parameters
        for (int i = 0; i < 3; ++i) {
            x_plus_delta[i] = x[i] + delta[i];
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool BiasParameterization::ComputeJacobian(const double* x,
                                          double* jacobian) const {
    // For Euclidean 3D bias, the Jacobian of Plus operation w.r.t delta is Identity
    // d(x + delta)/d(delta) = I (3x3 identity matrix)
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac(jacobian);
    jac.setIdentity();
    return true;
}

// ===============================================================================
// GRAVITY PARAMETERIZATION IMPLEMENTATION (ORB-SLAM3 Style)
// ===============================================================================

Eigen::Matrix3d GravityParameterization::ExpSO3(const Eigen::Vector3d& w) {
    return ExpSO3(w[0], w[1], w[2]);
}

Eigen::Matrix3d GravityParameterization::ExpSO3(double x, double y, double z) {
    // Rodrigues formula: exp([w]×) = I + sin(θ)/θ [w]× + (1-cos(θ))/θ² [w]×²
    // where θ = ||w|| and [w]× is the skew-symmetric matrix
    
    const double d2 = x*x + y*y + z*z;
    const double d = std::sqrt(d2);
    
    // Skew-symmetric matrix [w]×
    Eigen::Matrix3d W;
    W << 0.0, -z,   y,
         z,   0.0, -x,
        -y,   x,   0.0;
    
    Eigen::Matrix3d res;
    if (d < 1e-5) {
        // Small angle approximation: exp([w]×) ≈ I + [w]× + 0.5*[w]×²
        res = Eigen::Matrix3d::Identity() + W + 0.5*W*W;
    } else {
        // Full Rodrigues formula
        res = Eigen::Matrix3d::Identity() + W*std::sin(d)/d + W*W*(1.0 - std::cos(d))/d2;
    }
    
    return NormalizeRotation(res);
}

Eigen::Matrix3d GravityParameterization::NormalizeRotation(const Eigen::Matrix3d& R) {
    // Normalize rotation matrix using SVD to ensure orthogonality
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R_normalized = svd.matrixU() * svd.matrixV().transpose();
    
    // Ensure det(R) = +1 (proper rotation, not reflection)
    if (R_normalized.determinant() < 0) {
        Eigen::Matrix3d V_corrected = svd.matrixV();
        V_corrected.col(2) *= -1;  // Flip last column
        R_normalized = svd.matrixU() * V_corrected.transpose();
    }
    
    return R_normalized;
}

bool GravityParameterization::Plus(const double* x,
                                   const double* delta,
                                   double* x_plus_delta) const {
    try {
        // If parameter is fixed, don't apply any updates
        if (m_is_fixed) {
            for (int i = 0; i < 9; ++i) {
                x_plus_delta[i] = x[i];
            }
            return true;
        }
        
        // Reconstruct Rwg from column-major array [9 values]
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> Rwg(x);
        
        // Apply perturbation: Rwg_new = Rwg * ExpSO3(delta[0], delta[1], 0.0)
        // Note: Z-axis rotation is constrained to 0 for gravity direction
        Eigen::Vector3d perturbation(delta[0], delta[1], 0.0);
        Eigen::Matrix3d dR = ExpSO3(perturbation);
        
        Eigen::Matrix3d Rwg_new = Rwg * dR;
        
        // Store result in column-major order
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> result(x_plus_delta);
        result = Rwg_new;
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool GravityParameterization::ComputeJacobian(const double* x,
                                              double* jacobian) const {
    // Jacobian of Rwg_new = Rwg * ExpSO3(delta[0], delta[1], 0) w.r.t delta
    // 
    // For small perturbations, ExpSO3(ε) ≈ I + [ε]×
    // So: Rwg_new ≈ Rwg * (I + [ε]×) = Rwg + Rwg*[ε]×
    //
    // The Jacobian ∂(Rwg_new)/∂ε has size [9 x 2]
    // Each column corresponds to derivative w.r.t delta[0] and delta[1]
    
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> Rwg(x);
    Eigen::Map<Eigen::Matrix<double, 9, 2, Eigen::RowMajor>> jac(jacobian);
    
    // ∂(Rwg * ExpSO3(e1, e2, 0))/∂e1 evaluated at e1=e2=0
    // = Rwg * ∂(ExpSO3(e1, e2, 0))/∂e1
    // = Rwg * [∂([e]×)/∂e1] at e=[e1, e2, 0]
    //
    // [e]× = [ 0  -0   e2 ]     ∂([e]×)/∂e1 = [ 0  0  0 ]
    //        [ 0   0  -e1 ]                     [ 0  0 -1 ]
    //        [-e2  e1  0  ]                     [ 0  1  0 ]
    //
    // So: ∂(Rwg*ExpSO3)/∂e1 = Rwg * G1 where G1 is the generator for e1
    
    // Generator for delta[0] (perturbation around Y-axis)
    Eigen::Matrix3d G1;
    G1 << 0, 0,  0,
          0, 0, -1,
          0, 1,  0;
    
    // Generator for delta[1] (perturbation around X-axis)
    Eigen::Matrix3d G2;
    G2 << 0,  0, 1,
          0,  0, 0,
         -1,  0, 0;
    
    // Jacobian columns: vectorize(Rwg * G1) and vectorize(Rwg * G2)
    Eigen::Matrix3d J1 = Rwg * G1;
    Eigen::Matrix3d J2 = Rwg * G2;
    
    // Store in row-major format [9x2]
    jac.col(0) = Eigen::Map<Eigen::VectorXd>(J1.data(), 9);
    jac.col(1) = Eigen::Map<Eigen::VectorXd>(J2.data(), 9);
    
    return true;
}

} // namespace factor
} // namespace lightweight_vio
