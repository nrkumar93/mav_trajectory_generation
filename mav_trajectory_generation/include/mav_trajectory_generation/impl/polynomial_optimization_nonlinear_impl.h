/*
* Copyright (c) 2015, Markus Achtelik, ASL, ETH Zurich, Switzerland
* You can contact the author at <markus dot achtelik at mavt dot ethz dot ch>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef MAV_TRAJECTORY_GENERATION_IMPL_POLYNOMIAL_OPTIMIZATION_NONLINEAR_IMPL_H_
#define MAV_TRAJECTORY_GENERATION_IMPL_POLYNOMIAL_OPTIMIZATION_NONLINEAR_IMPL_H_

#include <chrono>
#include <numeric>

#include "mav_trajectory_generation/polynomial_optimization_linear.h"
#include "mav_trajectory_generation/timing.h"

namespace mav_trajectory_generation {

inline std::ostream& operator<<(std::ostream& stream,
                                const OptimizationInfo& val) {
  stream << "--- optimization info ---" << std::endl;
  stream << "  optimization time:     " << val.optimization_time << std::endl;
  stream << "  n_iterations:          " << val.n_iterations << std::endl;
  stream << "  stopping reason:       "
         << nlopt::returnValueToString(val.stopping_reason) << std::endl;
  stream << "  cost trajectory:       " << val.cost_trajectory << std::endl;
  stream << "  cost time:             " << val.cost_time << std::endl;
  stream << "  cost soft constraints: " << val.cost_soft_constraints
         << std::endl;
  stream << "  maxima: " << std::endl;
  for (const std::pair<int, Extremum>& m : val.maxima) {
    stream << "    " << positionDerivativeToString(m.first) << ": "
           << m.second.value << " in segment " << m.second.segment_idx
           << " and segment time " << m.second.time << std::endl;
  }
  return stream;
}

template <int _N>
PolynomialOptimizationNonLinear<_N>::PolynomialOptimizationNonLinear(
    size_t dimension, const NonlinearOptimizationParameters& parameters)
    : poly_opt_(dimension), optimization_parameters_(parameters) {}

template <int _N>
bool PolynomialOptimizationNonLinear<_N>::setupFromVertices(
    const Vertex::Vector& vertices, const std::vector<double>& segment_times,
    int derivative_to_optimize) {
  bool ret = poly_opt_.setupFromVertices(vertices, segment_times,
                                         derivative_to_optimize);

  size_t n_optimization_parameters;
  switch (optimization_parameters_.time_alloc_method) {
    case NonlinearOptimizationParameters::kSquaredTime:
    case NonlinearOptimizationParameters::kRichterTime:
    case NonlinearOptimizationParameters::kMellingerOuterLoop:
      n_optimization_parameters = segment_times.size();
      break;
    default:
      n_optimization_parameters =
          segment_times.size() +
          poly_opt_.getNumberFreeConstraints() * poly_opt_.getDimension();
      break;
  }

  nlopt_.reset(new nlopt::opt(optimization_parameters_.algorithm,
                              n_optimization_parameters));
  nlopt_->set_ftol_rel(optimization_parameters_.f_rel);
  nlopt_->set_ftol_abs(optimization_parameters_.f_abs);
  nlopt_->set_xtol_rel(optimization_parameters_.x_rel);
  nlopt_->set_xtol_abs(optimization_parameters_.x_abs);
  nlopt_->set_maxeval(optimization_parameters_.max_iterations);

  if (optimization_parameters_.random_seed < 0)
    nlopt_srand_time();
  else
    nlopt_srand(optimization_parameters_.random_seed);

  return ret;
}

template <int _N>
bool PolynomialOptimizationNonLinear<_N>::solveLinear() {
  return poly_opt_.solveLinear();
}

template <int _N>
int PolynomialOptimizationNonLinear<_N>::optimize() {

  ///> By Ram: optimization with collision
  traj_trace_.clear();

  optimization_info_ = OptimizationInfo();
  int result = nlopt::FAILURE;

  const std::chrono::high_resolution_clock::time_point t_start =
      std::chrono::high_resolution_clock::now();

  switch (optimization_parameters_.time_alloc_method) {
    case NonlinearOptimizationParameters::kSquaredTime:
    case NonlinearOptimizationParameters::kRichterTime:
      result = optimizeTime();
      break;
    case NonlinearOptimizationParameters::kSquaredTimeAndConstraints:
    case NonlinearOptimizationParameters::kRichterTimeAndConstraints:
      result = optimizeTimeAndFreeConstraints();
      break;
    /// By Ram: Old GD stuff
    case NonlinearOptimizationParameters::kRichterTimeAndConstraintsGD:
      result = optimizeTimeAndFreeConstraintsRichterGD();
      break;
    case NonlinearOptimizationParameters::kMellingerOuterLoop:
      result = optimizeTimeMellingerOuterLoop();
      break;
    default:
      break;
  }

  const std::chrono::high_resolution_clock::time_point t_stop =
      std::chrono::high_resolution_clock::now();
  optimization_info_.optimization_time =
      std::chrono::duration_cast<std::chrono::duration<double> >(t_stop -
                                                                 t_start)
          .count();

  optimization_info_.stopping_reason = result;

  return result;
}

template <int _N>
int PolynomialOptimizationNonLinear<_N>::optimizeTime() {
  std::vector<double> initial_step, segment_times;

  poly_opt_.getSegmentTimes(&segment_times);
  const size_t n_segments = segment_times.size();

  initial_step.reserve(n_segments);
  for (double t : segment_times) {
    initial_step.push_back(optimization_parameters_.initial_stepsize_rel * t);
  }

  try {
    // Set a lower bound on the segment time per segment to avoid numerical
    // issues.
    nlopt_->set_initial_step(initial_step);
    nlopt_->set_upper_bounds(std::numeric_limits<double>::max());
    nlopt_->set_lower_bounds(kOptimizationTimeLowerBound);
    nlopt_->set_min_objective(
        &PolynomialOptimizationNonLinear<N>::objectiveFunctionTime, this);

//    ///> By Ram: Testing nonlinear constraint
//    nlopt_->add_inequality_constraint(&PolynomialOptimizationNonLinear<N>::nonlinearConstraintTest, this);

  } catch (std::exception& e) {
    LOG(ERROR) << "error while setting up nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  double final_cost = std::numeric_limits<double>::max();
  int result;

  try {
    result = nlopt_->optimize(segment_times, final_cost);
  } catch (std::exception& e) {
    LOG(ERROR) << "error while running nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  return result;
}

template <int _N>
int PolynomialOptimizationNonLinear<_N>::optimizeTimeMellingerOuterLoop() {
  std::vector<double> segment_times;
  poly_opt_.getSegmentTimes(&segment_times);

  // Save original segment times
  std::vector<double> original_segment_times = segment_times;

  if (optimization_parameters_.print_debug_info_time_allocation) {
    std::cout << "Segment times: ";
    for (const double seg_time : segment_times) {
      std::cout << seg_time << " ";
    }
    std::cout << std::endl;
  }

  try {
    // Set a lower bound on the segment time per segment to avoid numerical
    // issues.
    nlopt_->set_upper_bounds(std::numeric_limits<double>::max());
    nlopt_->set_lower_bounds(kOptimizationTimeLowerBound);
    nlopt_->set_min_objective(&PolynomialOptimizationNonLinear<
                                  N>::objectiveFunctionTimeMellingerOuterLoop,
                              this);
  } catch (std::exception& e) {
    LOG(ERROR) << "error while setting up nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  double final_cost = std::numeric_limits<double>::max();
  int result = nlopt::FAILURE;

  try {
    result = nlopt_->optimize(segment_times, final_cost);
  } catch (std::exception& e) {
    LOG(ERROR) << "error while running nlopt: " << e.what()
               << ". This likely means the optimization aborted early."
               << std::endl;
    if (final_cost == std::numeric_limits<double>::max()) {
      return nlopt::FAILURE;
    }

    if (optimization_parameters_.print_debug_info_time_allocation) {
      std::cout << "Segment times after opt: ";
      for (const double seg_time : segment_times) {
        std::cout << seg_time << " ";
      }
      std::cout << std::endl;
      std::cout << "Final cost: " << final_cost << std::endl;
      std::cout << "Nlopt result: " << result << std::endl;
    }
  }

  // Scaling of segment times
  std::vector<double> relative_segment_times;
  poly_opt_.getSegmentTimes(&relative_segment_times);
  scaleSegmentTimesWithViolation();
  std::vector<double> scaled_segment_times;
  poly_opt_.getSegmentTimes(&scaled_segment_times);

  // Print all parameter after scaling
  if (optimization_parameters_.print_debug_info_time_allocation) {
    std::cout << "[MEL          Original]: ";
    std::for_each(original_segment_times.cbegin(),
                  original_segment_times.cend(),
                  [](double c) { std::cout << c << " "; });
    std::cout << std::endl;
    std::cout << "[MEL RELATIVE Solution]: ";
    std::for_each(relative_segment_times.cbegin(),
                  relative_segment_times.cend(),
                  [](double c) { std::cout << c << " "; });
    std::cout << std::endl;
    std::cout << "[MEL          Solution]: ";
    std::for_each(scaled_segment_times.cbegin(), scaled_segment_times.cend(),
                  [](double c) { std::cout << c << " "; });
    std::cout << std::endl;
    std::cout << "[MEL   Trajectory Time] Before: "
              << std::accumulate(original_segment_times.begin(),
                                 original_segment_times.end(), 0.0)
              << " | After Rel Change: "
              << std::accumulate(relative_segment_times.begin(),
                                 relative_segment_times.end(), 0.0)
              << " | After Scaling: "
              << std::accumulate(scaled_segment_times.begin(),
                                 scaled_segment_times.end(), 0.0)
              << std::endl;
  }

  return result;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::getCost() const {
  return poly_opt_.computeCost();
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::getTotalCostWithSoftConstraints()
    const {
  double cost_trajectory = poly_opt_.computeCost();

  // Use consistent cost metrics regardless of method set, to compare between
  // methods.
  std::vector<double> segment_times;
  poly_opt_.getSegmentTimes(&segment_times);
  double total_time =
      std::accumulate(segment_times.begin(), segment_times.end(), 0.0);
  double cost_time =
      total_time * total_time * optimization_parameters_.time_penalty;
  double cost_constraints = evaluateMaximumMagnitudeAsSoftConstraint(
      inequality_constraints_, optimization_parameters_.soft_constraint_weight,
      1e9);

  return cost_trajectory + cost_time + cost_constraints;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::getCostAndGradientMellinger(
    std::vector<double>* gradients) {
  // Weighting terms for different costs
  // Retrieve the current segment times
  std::vector<double> segment_times;
  poly_opt_.getSegmentTimes(&segment_times);
  const double J_d = poly_opt_.computeCost();

  if (poly_opt_.getNumberSegments() == 1) {
    if (gradients != NULL) {
      gradients->clear();
      gradients->resize(poly_opt_.getNumberSegments(), 0.0);
    }

    return J_d;
  }

  if (gradients != NULL) {
    const size_t n_segments = poly_opt_.getNumberSegments();

    gradients->clear();
    gradients->resize(n_segments);

    // Initialize changed segment times for numerical derivative
    std::vector<double> segment_times_bigger(n_segments);
    const double increment_time = 0.1;
    for (size_t n = 0; n < n_segments; ++n) {
      // Now the same with an increased segment time
      // Calculate cost with higher segment time
      segment_times_bigger = segment_times;
      // Deduct h*(-1/(m-2)) according to paper Mellinger "Minimum snap
      // trajectory generation and control for quadrotors"
      double const_traj_time_corr = increment_time / (n_segments - 1.0);
      for (size_t i = 0; i < segment_times_bigger.size(); ++i) {
        if (i == n) {
          segment_times_bigger[i] += increment_time;
        } else {
          segment_times_bigger[i] -= const_traj_time_corr;
        }
      }

      // TODO: add case if segment_time is at threshold 0.1s
      // 1) How many segments > 0.1s
      // 2) trajectory time correction only on those
      // for (int j = 0; j < segment_times_bigger.size(); ++j) {
      //   double thresh_corr = 0.0;
      //   if (segment_times_bigger[j] < 0.1) {
      //     thresh_corr = 0.1-segment_times_bigger[j];
      //   }
      // }

      // Check and make sure that segment times are >
      // kOptimizationTimeLowerBound
      for (double& t : segment_times_bigger) {
        t = std::max(kOptimizationTimeLowerBound, t);
      }

      // Update the segment times. This changes the polynomial coefficients.
      poly_opt_.updateSegmentTimes(segment_times_bigger);
      poly_opt_.solveLinear();

      // Calculate cost and gradient with new segment time
      const double J_d_bigger = poly_opt_.computeCost();
      const double dJd_dt = (J_d_bigger - J_d) / increment_time;

      // Calculate the gradient
      gradients->at(n) = dJd_dt;
    }

    // Set again the original segment times from before calculating the
    // numerical gradient
    poly_opt_.updateSegmentTimes(segment_times);
    poly_opt_.solveLinear();
  }

  // Compute cost without gradient
  return J_d;
}

template <int _N>
void PolynomialOptimizationNonLinear<_N>::scaleSegmentTimesWithViolation() {
  // Get trajectory
  Trajectory traj;
  poly_opt_.getTrajectory(&traj);

  // Get constraints
  double v_max = 0.0;
  double a_max = 0.0;
  for (const auto& constraint : inequality_constraints_) {
    if (constraint->derivative == derivative_order::VELOCITY) {
      v_max = constraint->value;
    } else if (constraint->derivative == derivative_order::ACCELERATION) {
      a_max = constraint->value;
    }
  }

  if (optimization_parameters_.print_debug_info_time_allocation) {
    double v_max_actual, a_max_actual;
    traj.computeMaxVelocityAndAcceleration(&v_max_actual, &a_max_actual);
    std::cout << "[Time Scaling] Beginning:  v: max: " << v_max_actual << " / "
              << v_max << " a: max: " << a_max_actual << " / " << a_max
              << std::endl;
  }

  // Run the trajectory time scaling.
  traj.scaleSegmentTimesToMeetConstraints(v_max, a_max);

  std::vector<double> segment_times;
  segment_times = traj.getSegmentTimes();
  poly_opt_.updateSegmentTimes(segment_times);
  poly_opt_.solveLinear();

  if (optimization_parameters_.print_debug_info_time_allocation) {
    double v_max_actual, a_max_actual;
    traj.computeMaxVelocityAndAcceleration(&v_max_actual, &a_max_actual);
    std::cout << "[Time Scaling] End: v: max: " << v_max_actual << " / "
              << v_max << " a: max: " << a_max_actual << " / " << a_max
              << std::endl;
  }
}

template <int _N>
void PolynomialOptimizationNonLinear<_N>::setInitialGuess(std::vector<double> &init_guess) {
  initial_guess_ = init_guess;
}

template <int _N>
int PolynomialOptimizationNonLinear<_N>::optimizeTimeAndFreeConstraints() {
  std::vector<double> initial_step, initial_solution, segment_times,
      lower_bounds, upper_bounds;

  poly_opt_.getSegmentTimes(&segment_times);
  const size_t n_segments = segment_times.size();

  // compute initial solution
  poly_opt_.solveLinear();
  std::vector<Eigen::VectorXd> free_constraints;
  poly_opt_.getFreeConstraints(&free_constraints);
  if (free_constraints.size() == 0 || free_constraints.front().size() == 0) {
    LOG(WARNING)
        << "No free derivative variables, same as time-only optimization.";
  }

  const size_t n_optimization_variables =
      n_segments + free_constraints.size() * free_constraints.front().size();

  CHECK_GT(n_optimization_variables, 0u);

  initial_solution.reserve(n_optimization_variables);
  initial_step.reserve(n_optimization_variables);
  lower_bounds.reserve(n_optimization_variables);
  upper_bounds.reserve(n_optimization_variables);

  ///> By Ram: using the init guess
  // copy all constraints into one vector:
  if (initial_guess_.empty()) {
    for (double t : segment_times) {
      initial_solution.push_back(t);
    }

    for (const Eigen::VectorXd& c : free_constraints) {
      for (int i = 0; i < c.size(); ++i) {
        initial_solution.push_back(c[i]);
      }
    }
  } else {
    initial_solution = std::move(initial_guess_);
  }


  // Setup for getting bounds on the free endpoint derivatives
  std::vector<double> lower_bounds_free, upper_bounds_free;
  const size_t n_optimization_variables_free =
      free_constraints.size() * free_constraints.front().size();
  lower_bounds_free.reserve(n_optimization_variables_free);
  upper_bounds_free.reserve(n_optimization_variables_free);

  // Get the lower and upper bounds constraints on the free endpoint
  // derivatives
  Vertex::Vector vertices;
  poly_opt_.getVertices(&vertices);
  setFreeEndpointDerivativeHardConstraints(vertices, &lower_bounds_free,
                                           &upper_bounds_free);

  // Set segment time constraints
  for (size_t l = 0; l < n_segments; ++l) {
    lower_bounds.push_back(kOptimizationTimeLowerBound);
    upper_bounds.push_back(std::numeric_limits<double>::max());
  }
  // Append free endpoint derivative constraints
  lower_bounds.insert(std::end(lower_bounds), std::begin(lower_bounds_free),
                      std::end(lower_bounds_free));
  upper_bounds.insert(std::end(upper_bounds), std::begin(upper_bounds_free),
                      std::end(upper_bounds_free));

  for (size_t i = 0; i < initial_solution.size(); i++) {
    double x = initial_solution[i];
    const double abs_x = std::abs(x);
    // Initial step size cannot be 0.0 --> invalid arg
    if (abs_x <= std::numeric_limits<double>::lowest()) {
      initial_step.push_back(1e-13);
    } else {
      initial_step.push_back(optimization_parameters_.initial_stepsize_rel *
                             abs_x);
    }

    // Check if initial solution isn't already out of bounds.
    if (x < lower_bounds[i]) {
      lower_bounds[i] = x;
    } else if (x > upper_bounds[i]) {
      upper_bounds[i] = x;
    }
  }

  // Make sure everything is the same size, otherwise NLOPT will have a bad
  // time.
  CHECK_EQ(lower_bounds.size(), upper_bounds.size());
  CHECK_EQ(initial_solution.size(), lower_bounds.size());
  CHECK_EQ(initial_solution.size(), initial_step.size());
  CHECK_EQ(initial_solution.size(), n_optimization_variables);

  try {
    nlopt_->set_initial_step(initial_step);
    nlopt_->set_lower_bounds(lower_bounds);
    nlopt_->set_upper_bounds(upper_bounds);
    nlopt_->set_min_objective(&PolynomialOptimizationNonLinear<
                                  N>::objectiveFunctionTimeAndConstraints,
                              this);
  } catch (std::exception& e) {
    LOG(ERROR) << "error while setting up nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  double final_cost = std::numeric_limits<double>::max();
  int result;

  try {
    timing::Timer timer_solve("optimize_nonlinear_full_total_time");

    result = nlopt_->optimize(initial_solution, final_cost);
    timer_solve.Stop();
  } catch (std::exception& e) {
    LOG(ERROR) << "error while running nlopt: " << e.what() << std::endl;
    return nlopt::FAILURE;
  }

  ///> By Ram:
  initial_guess_.clear();

  return result;
}

/// By Ram: Old GD stuff
template <int _N>
int PolynomialOptimizationNonLinear<_N>::
optimizeTimeAndFreeConstraintsRichterGD() {
  const size_t n_segments = poly_opt_.getNumberSegments();
  const size_t n_free_constraints = poly_opt_.getNumberFreeConstraints();
  const size_t dim = poly_opt_.getDimension();

  // Get initial segment times.
  std::vector<double> segment_times;
  poly_opt_.getSegmentTimes(&segment_times);
  poly_opt_.solveLinear();

  // Create parameter vector x=[t1, ..., tm, dx1, ... dxv, dy, dz]
  Eigen::VectorXd x;
  x.resize(segment_times.size()+dim*n_free_constraints);
  if (initial_guess_.empty()) {
    for (size_t m = 0; m < n_segments; ++m) {
      x[m] = segment_times[m];
    }
    // Retrieve free constraints
    std::vector<Eigen::VectorXd> d_p_vec;
    poly_opt_.getFreeConstraints(&d_p_vec);
    // Append free constraints to parameter vector x
    for (int k = 0; k < dim; ++k) {
      x.block(n_segments+k*n_free_constraints, 0, n_free_constraints, 1) =
              d_p_vec[k];
    }
  } else {
    for (int i=0; i<initial_guess_.size(); ++i)
    {
      x[i] = initial_guess_[i];
    }
  }
  initial_guess_.clear();

  // Save original parameter vector
  Eigen::VectorXd x_orig;
  x_orig = x;

  // Set up gradients (of param vector x) and increment vector
  Eigen::VectorXd grad, increment;
  grad.resize(x.size());
  grad.setZero();
  increment = grad;

  // Weights for cost terms
  const double w_d = 0.1;
  const double w_t = 5.0;
  const double w_sc = 1.0;

  // Gradients for individual const terms
  std::vector<double> grad_t;
  std::vector<Eigen::VectorXd> grad_d, grad_sc;
  grad_d.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
  grad_sc.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));

  // Parameter for gradient descent
  /// @TODO: Exit when the obj func does not change beyond FTOL
  /// @TODO: Perhaps check and implement backtracking
  int max_iter = 50;
  double lambda = 10.0*(2.0+0.1); // TODO: Which value? // TODO: parameterize

  double prev_total_cost = std::numeric_limits<double>::infinity();
  double total_cost = 0;
  int stop_count = 0;


  double J_t = 0.0;
  double J_d = 0.0;
  double J_sc = 0.0;
  for (int i = 0; i < max_iter; ++i) {
//    std::cout << "GD Iter#: " << i << std::endl;
    // Evaluate cost.
    J_t = getCostAndGradientTimeForward(&grad_t);
    J_d = getCostAndGradientDerivative(&grad_d);
    J_sc = getCostAndGradientSoftConstraintsForward(&grad_sc);

    // Unpack gradients.
    for (int j = 0; j < n_segments; ++j) {
      grad[j] = grad_t[j];
    }
    for (int k = 0; k < dim; ++k) {
      const int start_idx = n_segments + (k * n_free_constraints);
      for (int i = 0; i < n_free_constraints; ++i) {
        grad[start_idx + i] = w_d * grad_d[k][i] + w_sc * grad_sc[k][i];
      }
    }

    double step_size = 1.0 / (lambda + i);
    increment = -step_size * grad;

    // Update the parameters.
    x += increment;
    // Check and make sure that segment times are > kOptimizationTimeLowerBound
    for (int n = 0; n < n_segments; ++n) {
      x[n] = std::max(kOptimizationTimeLowerBound, x[n]);
    }

    // Set new segment times and new free constraints
    std::vector<double> segment_times_new;
    segment_times_new.reserve(n_segments);
    for (size_t i = 0; i < n_segments; ++i) {
      segment_times_new.push_back(x[i]);
    }
    std::vector<Eigen::VectorXd> d_p_vec_new;
    d_p_vec_new.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
    for (int k = 0; k < dim; ++k) {
      d_p_vec_new[k] = x.block(n_segments+k*n_free_constraints, 0,
                               n_free_constraints, 1);
    }

    // Update segement times and free constraints
    poly_opt_.updateSegmentTimes(segment_times_new);
    poly_opt_.setFreeConstraints(d_p_vec_new);
    poly_opt_.solveLinear();

    double cost_time = computeTotalTrajectoryTime(segment_times_new) * optimization_parameters_.time_penalty;
    double cost_trajectory = poly_opt_.computeCost();
    prev_total_cost = total_cost;
    total_cost = cost_time + cost_trajectory;

    if (abs(total_cost - prev_total_cost) < optimization_parameters_.f_rel)
    {
      ++stop_count;
      if (stop_count > 5)
      {
        // Print only segment times
        if (optimization_parameters_.print_debug_info_time_allocation)
        {
          std::cout << "[GD RICHTER Original]: "
                    << x_orig.block(0, 0, n_segments, 1).transpose()
                    << std::endl;
          std::cout << "[GD RICHTER Solution]: "
                    << x.block(0, 0, n_segments, 1).transpose()
                    << std::endl;
          std::cout << "[GD RICHTER Trajectory Time] Before: "
                    << x_orig.block(0, 0, n_segments, 1).sum()
                    << " | After: " << x.block(0, 0, n_segments, 1).sum()
                    << std::endl;
        }
        std::cout << "GD Exit Iter#: " << i << std::endl;
        return nlopt::SUCCESS;
      }
      continue;
    }
    stop_count = 0;
  }

  std::cout << "GD Exit with FAILURE " << std::endl;
  return nlopt::FAILURE;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::getCostAndGradientTimeForward(
        std::vector<double>* gradients) {

  // Weighting terms for different costs
  const double w_d = 0.1;
  const double w_t = 5.0;
  const double w_sc = 1.0;

  // Retrieve the current segment times
  std::vector<double> segment_times;
  poly_opt_.getSegmentTimes(&segment_times);

  // Calculate current cost
  // TODO: parse from outside?
  // According to paper the endpoint derivative cost is cost = c^T * Q * c
  const double J_d = poly_opt_.computeCost();
  const double J_sc = getCostAndGradientSoftConstraintsForward(NULL);

  if (gradients != NULL) {
    const size_t n_segments = poly_opt_.getNumberSegments();

    gradients->clear();
    gradients->resize(n_segments);

    // Initialize changed segment times for numerical derivative
    std::vector<double> segment_times_bigger(n_segments);
    const double increment_time = 0.1;
    for (int n = 0; n < n_segments; ++n) {
      // Now the same with an increased segment time
      // Calculate cost with higher segment time
      segment_times_bigger = segment_times;
      // Check and make sure that segment times are >
      // kOptimizationTimeLowerBound, otherwise add increment_time to seg time
      segment_times_bigger[n] = std::max(kOptimizationTimeLowerBound,
                                         segment_times_bigger[n] + increment_time);

      // Update the segment times. This changes the polynomial coefficients.
      poly_opt_.updateSegmentTimes(segment_times_bigger);
      poly_opt_.solveLinear();

      // Calculate cost and gradient with new segment time
      const double J_d_bigger = poly_opt_.computeCost();
      double J_sc_bigger = 0.0;
      if (optimization_parameters_.use_soft_constraints) {
        J_sc_bigger = getCostAndGradientSoftConstraintsForward(NULL);
      }

      const double dJd_dt = (J_d_bigger-J_d) / (increment_time);
      const double dJsc_dt = (J_sc_bigger-J_sc) / (increment_time);
      // TODO: also for cost_time_method = kSquared
      const double dJt_dt = 1.0; // J_t = t --> dJt_dt = 1.0 for all tm

      // Calculate the gradient
      if (optimization_parameters_.use_soft_constraints) {
        gradients->at(n) = w_d * dJd_dt + w_sc * dJsc_dt + w_t * dJt_dt;
      } else {
        gradients->at(n) = w_d * dJd_dt + w_t * dJt_dt;
      }
    }

    // Set again the original segment times from before calculating the
    // numerical gradient
    poly_opt_.updateSegmentTimes(segment_times);
    poly_opt_.solveLinear();
  }

  // Compute cost without gradient (only time)
  double total_time = computeTotalTrajectoryTime(segment_times);
  double J_t = total_time;  // TODO: Distinguish Richter vs own (squared)

  return J_t;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::getCostAndGradientDerivative(
        std::vector<Eigen::VectorXd>* gradients) {

  // Compare the two approaches: getCost() and the full matrix.
  const size_t n_free_constraints = poly_opt_.getNumberFreeConstraints();
  const size_t n_fixed_constraints = poly_opt_.getNumberFixedConstraints();
  const size_t dim = poly_opt_.getDimension();

  double J_d = 0.0;
  std::vector<Eigen::VectorXd> grad_d(
          dim, Eigen::VectorXd::Zero(n_free_constraints));

  // Retrieve R
  Eigen::MatrixXd R;
  poly_opt_.getR(&R);

  // Set up mappings to R_FF R_FP R_PP etc. R_FP' = R_PF if that saves
  // time eventually.
  // All of these are the same per axis.
  // R_ff * d_f is actually constant so can cache this term.
  const Eigen::Block<Eigen::MatrixXd> R_ff =
          R.block(0, 0, n_fixed_constraints, n_fixed_constraints);
  const Eigen::Block<Eigen::MatrixXd> R_pf =
          R.block(n_fixed_constraints, 0, n_free_constraints,
                  n_fixed_constraints);
  const Eigen::Block<Eigen::MatrixXd> R_pp =
          R.block(n_fixed_constraints, n_fixed_constraints, n_free_constraints,
                  n_free_constraints);

  // Get d_p and d_f vector for all axes.
  std::vector<Eigen::VectorXd> d_p_vec;
  std::vector<Eigen::VectorXd> d_f_vec;
  poly_opt_.getFreeConstraints(&d_p_vec);
  poly_opt_.getFixedConstraints(&d_f_vec);

  Eigen::MatrixXd J_d_temp;
  // Compute costs over all axes.
  for (int k = 0; k < dim; ++k) {
    // Get a copy of d_p and d_f for this axis.
    const Eigen::VectorXd& d_p = d_p_vec[k];
    const Eigen::VectorXd& d_f = d_f_vec[k];

    // Now do the other thing.
    J_d_temp = d_f.transpose() * R_ff * d_f +
               d_f.transpose() * R_pf.transpose() * d_p +
               d_p.transpose() * R_pf * d_f + d_p.transpose() * R_pp * d_p;
    J_d += J_d_temp(0, 0);

    // And get the gradient.
    // Should really separate these out by k.
    grad_d[k] =
            2 * d_f.transpose() * R_pf.transpose() + 2 * d_p.transpose() * R_pp;
  }

  if (gradients != NULL) {
    gradients->clear();
    gradients->resize(dim);

    for (int k = 0; k < dim; ++k) {
      (*gradients)[k] = grad_d[k];
    }
  }

  return J_d;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::
getCostAndGradientSoftConstraintsForward(
        std::vector<Eigen::VectorXd>* gradients) {

  double J_sc = evaluateMaximumMagnitudeAsSoftConstraint(
          inequality_constraints_,
          optimization_parameters_.soft_constraint_weight);

  if (gradients != NULL) {
    const size_t n_free_constraints = poly_opt_.getNumberFreeConstraints();
    const size_t dim = poly_opt_.getDimension();

    gradients->clear();
    gradients->resize(dim, Eigen::VectorXd::Zero(n_free_constraints));

    // Get the current free constraints
    std::vector<Eigen::VectorXd> free_constraints;
    poly_opt_.getFreeConstraints(&free_constraints);

    std::vector<Eigen::VectorXd> free_constraints_right;
    free_constraints_right.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));
    const double increment_dist = 0.05;

    std::vector<Eigen::VectorXd> increment(
            dim, Eigen::VectorXd::Zero(n_free_constraints));
    for (int k = 0; k < dim; ++k) {
      increment.clear();
      increment.resize(dim, Eigen::VectorXd::Zero(n_free_constraints));

      for (int n = 0; n < n_free_constraints; ++n) {
        increment[k].setZero();
        increment[k][n] = increment_dist;

        for (int k2 = 0; k2 < dim; ++k2) {
          free_constraints_right[k2] = free_constraints[k2] + increment[k2];
        }
        poly_opt_.setFreeConstraints(free_constraints_right);
        const double cost_right =
                evaluateMaximumMagnitudeAsSoftConstraint(
                        inequality_constraints_,
                        optimization_parameters_.soft_constraint_weight);

        const double grad_k_n = (cost_right - J_sc) / (increment_dist);
        gradients->at(k)[n] = grad_k_n;
      }
    }

    // Set again the original constraints from before calculating the numerical
    // constraints
    poly_opt_.setFreeConstraints(free_constraints);
  }

  return J_sc;
}

template <int _N>
bool PolynomialOptimizationNonLinear<_N>::addMaximumMagnitudeConstraint(
    int derivative, double maximum_value) {
  CHECK_GE(derivative, 0);
  CHECK_GE(maximum_value, 0.0);

  std::shared_ptr<ConstraintData> constraint_data(new ConstraintData);
  constraint_data->derivative = derivative;
  constraint_data->value = maximum_value;
  constraint_data->this_object = this;

  // Store the shared_ptrs such that their data will be destroyed later.
  inequality_constraints_.push_back(constraint_data);

  if (!optimization_parameters_.use_soft_constraints) {
    try {
      nlopt_->add_inequality_constraint(
          &PolynomialOptimizationNonLinear<
              N>::evaluateMaximumMagnitudeConstraint,
          constraint_data.get(),
          optimization_parameters_.inequality_constraint_tolerance);
    } catch (std::exception& e) {
      LOG(ERROR) << "ERROR while setting inequality constraint " << e.what()
                 << std::endl;
      return false;
    }
  }
  return true;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::objectiveFunctionTime(
    const std::vector<double>& segment_times, std::vector<double>& gradient,
    void* data) {
  CHECK(gradient.empty())
      << "computing gradient not possible, choose a gradient free method";
  CHECK_NOTNULL(data);

  PolynomialOptimizationNonLinear<N>* optimization_data =
      static_cast<PolynomialOptimizationNonLinear<N>*>(data);  // wheee ...

  CHECK_EQ(segment_times.size(),
           optimization_data->poly_opt_.getNumberSegments());

  optimization_data->poly_opt_.updateSegmentTimes(segment_times);
  optimization_data->poly_opt_.solveLinear();
  double cost_trajectory = optimization_data->poly_opt_.computeCost();
  double cost_time = 0;
  double cost_constraints = 0;
  const double total_time = computeTotalTrajectoryTime(segment_times);

  switch (optimization_data->optimization_parameters_.time_alloc_method) {
    case NonlinearOptimizationParameters::kRichterTime:
      cost_time =
          total_time * optimization_data->optimization_parameters_.time_penalty;
      break;
    default:  // kSquaredTime
      cost_time = total_time * total_time *
                  optimization_data->optimization_parameters_.time_penalty;
      break;
  }

  ///> By Ram: Adding code to log trajectory convergence trace
  mav_trajectory_generation::Trajectory curr_traj;
  optimization_data->getTrajectory(&curr_traj);
  optimization_data->traj_trace_.push_back(std::move(curr_traj));

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "---- cost at iteration "
              << optimization_data->optimization_info_.n_iterations << "---- "
              << std::endl;
    std::cout << "  trajectory: " << cost_trajectory << std::endl;
    std::cout << "  time: " << cost_time << std::endl;
  }

  if (optimization_data->optimization_parameters_.use_soft_constraints) {
    cost_constraints =
        optimization_data->evaluateMaximumMagnitudeAsSoftConstraint(
            optimization_data->inequality_constraints_,
            optimization_data->optimization_parameters_.soft_constraint_weight);
  }

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "  sum: " << cost_trajectory + cost_time + cost_constraints
              << std::endl;
    std::cout << "  total time: " << total_time << std::endl;
  }

  optimization_data->optimization_info_.n_iterations++;
  optimization_data->optimization_info_.cost_trajectory = cost_trajectory;
  optimization_data->optimization_info_.cost_time = cost_time;
  optimization_data->optimization_info_.cost_soft_constraints =
      cost_constraints;

  return cost_trajectory + cost_time + cost_constraints;
}

template <int _N>
double
PolynomialOptimizationNonLinear<_N>::objectiveFunctionTimeMellingerOuterLoop(
    const std::vector<double>& segment_times, std::vector<double>& gradient,
    void* data) {
  CHECK(!gradient.empty())
      << "only with gradients possible, choose a gradient based method";
  CHECK_NOTNULL(data);

  PolynomialOptimizationNonLinear<N>* optimization_data =
      static_cast<PolynomialOptimizationNonLinear<N>*>(data);  // wheee ...

  CHECK_EQ(segment_times.size(),
           optimization_data->poly_opt_.getNumberSegments());

  optimization_data->poly_opt_.updateSegmentTimes(segment_times);
  optimization_data->poly_opt_.solveLinear();
  double cost_trajectory;
  if (!gradient.empty()) {
    cost_trajectory = optimization_data->getCostAndGradientMellinger(&gradient);
  } else {
    cost_trajectory = optimization_data->getCostAndGradientMellinger(NULL);
  }

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "---- cost at iteration "
              << optimization_data->optimization_info_.n_iterations << "---- "
              << std::endl;
    std::cout << "  segment times: ";
    for (double segment_time : segment_times) {
      std::cout << segment_time << " ";
    }
    std::cout << std::endl;
    std::cout << "  sum: " << cost_trajectory << std::endl;
  }

  optimization_data->optimization_info_.n_iterations++;
  optimization_data->optimization_info_.cost_trajectory = cost_trajectory;

  return cost_trajectory;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::objectiveFunctionTimeAndConstraints(
    const std::vector<double>& x, std::vector<double>& gradient, void* data) {
  CHECK(gradient.empty())
      << "computing gradient not possible, choose a gradient-free method";
  CHECK_NOTNULL(data);

  PolynomialOptimizationNonLinear<N>* optimization_data =
      static_cast<PolynomialOptimizationNonLinear<N>*>(data);  // wheee ...

  const size_t n_segments = optimization_data->poly_opt_.getNumberSegments();
  const size_t n_free_constraints =
      optimization_data->poly_opt_.getNumberFreeConstraints();
  const size_t dim = optimization_data->poly_opt_.getDimension();

  CHECK_EQ(x.size(), n_segments + n_free_constraints * dim);

  std::vector<Eigen::VectorXd> free_constraints;
  free_constraints.resize(dim);
  std::vector<double> segment_times;
  segment_times.reserve(n_segments);

  for (size_t i = 0; i < n_segments; ++i) {
    segment_times.push_back(x[i]);
  }

  for (size_t d = 0; d < dim; ++d) {
    const size_t idx_start = n_segments + d * n_free_constraints;

    Eigen::VectorXd& free_constraints_dim = free_constraints[d];
    free_constraints_dim.resize(n_free_constraints, Eigen::NoChange);
    for (size_t i = 0; i < n_free_constraints; ++i) {
      free_constraints_dim[i] = x[idx_start + i];
    }
  }

  optimization_data->poly_opt_.updateSegmentTimes(segment_times);
  optimization_data->poly_opt_.setFreeConstraints(free_constraints);

  double cost_trajectory = optimization_data->poly_opt_.computeCost();
  double cost_time = 0;
  double cost_constraints = 0;

  const double total_time = computeTotalTrajectoryTime(segment_times);
  switch (optimization_data->optimization_parameters_.time_alloc_method) {
    case NonlinearOptimizationParameters::kRichterTimeAndConstraints:
      cost_time =
          total_time * optimization_data->optimization_parameters_.time_penalty;
      break;
    default:  // kSquaredTimeAndConstraints
      cost_time = total_time * total_time *
                  optimization_data->optimization_parameters_.time_penalty;
      break;
  }

  ///> By Ram: Adding code to log trajectory convergence trace
  mav_trajectory_generation::Trajectory curr_traj;
  optimization_data->getTrajectory(&curr_traj);
  optimization_data->traj_trace_.push_back(std::move(curr_traj));


  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "---- cost at iteration "
              << optimization_data->optimization_info_.n_iterations << "---- "
              << std::endl;
    std::cout << "  trajectory: " << cost_trajectory << std::endl;
    std::cout << "  time: " << cost_time << std::endl;
  }

  if (optimization_data->optimization_parameters_.use_soft_constraints) {
    cost_constraints =
        optimization_data->evaluateMaximumMagnitudeAsSoftConstraint(
            optimization_data->inequality_constraints_,
            optimization_data->optimization_parameters_.soft_constraint_weight);
  }

  if (optimization_data->optimization_parameters_.print_debug_info) {
    std::cout << "  sum: " << cost_trajectory + cost_time + cost_constraints
              << std::endl;
    std::cout << "  total time: " << total_time << std::endl;
  }

  optimization_data->optimization_info_.n_iterations++;
  optimization_data->optimization_info_.cost_trajectory = cost_trajectory;
  optimization_data->optimization_info_.cost_time = cost_time;
  optimization_data->optimization_info_.cost_soft_constraints =
      cost_constraints;

  return cost_trajectory + cost_time + cost_constraints;
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::evaluateMaximumMagnitudeConstraint(
    const std::vector<double>& segment_times, std::vector<double>& gradient,
    void* data) {
  CHECK(gradient.empty())
      << "computing gradient not possible, choose a gradient-free method";
  ConstraintData* constraint_data =
      static_cast<ConstraintData*>(data);  // wheee ...
  PolynomialOptimizationNonLinear<N>* optimization_data =
      constraint_data->this_object;

  Extremum max;
  max = optimization_data->poly_opt_.computeMaximumOfMagnitude(
      constraint_data->derivative, nullptr);

  optimization_data->optimization_info_.maxima[constraint_data->derivative] =
      max;

  return max.value - constraint_data->value;
}

template <int _N>
double
PolynomialOptimizationNonLinear<_N>::evaluateMaximumMagnitudeAsSoftConstraint(
    const std::vector<std::shared_ptr<ConstraintData> >& inequality_constraints,
    double weight, double maximum_cost) const {
  std::vector<double> dummy;
  double cost = 0;

  if (optimization_parameters_.print_debug_info)
    std::cout << "  soft_constraints: " << std::endl;

  for (std::shared_ptr<const ConstraintData> constraint :
       inequality_constraints_) {
    // need to call the c-style callback function here, thus the ugly cast to
    // void*.
    double abs_violation = evaluateMaximumMagnitudeConstraint(
        dummy, dummy, (void*)constraint.get());

    double relative_violation = abs_violation / constraint->value;
    const double current_cost =
        std::min(maximum_cost, exp(relative_violation * weight));
    cost += current_cost;
    if (optimization_parameters_.print_debug_info) {
      std::cout << "    derivative " << constraint->derivative
                << " abs violation: " << abs_violation
                << " : relative violation: " << relative_violation
                << " cost: " << current_cost << std::endl;
    }
  }
  return cost;
}

template <int _N>
void PolynomialOptimizationNonLinear<_N>::
    setFreeEndpointDerivativeHardConstraints(
        const Vertex::Vector& vertices, std::vector<double>* lower_bounds,
        std::vector<double>* upper_bounds) {
  CHECK_NOTNULL(lower_bounds);
  CHECK_NOTNULL(upper_bounds);
  CHECK(lower_bounds->empty()) << "Lower bounds not empty!";
  CHECK(upper_bounds->empty()) << "Upper bounds not empty!";

  const size_t n_free_constraints = poly_opt_.getNumberFreeConstraints();
  const size_t dim = poly_opt_.getDimension();
  const int derivative_to_optimize = poly_opt_.getDerivativeToOptimize();

  // Set all values to -inf/inf and reset only bounded opti param with values
  lower_bounds->resize(dim * n_free_constraints,
                       std::numeric_limits<double>::lowest());
  upper_bounds->resize(dim * n_free_constraints,
                       std::numeric_limits<double>::max());

  // Add higher order derivative constraints (v_max and a_max)
  // Check at each vertex which of the derivatives is a free derivative.
  // If it is a free derivative check if we have a constraint in
  // inequality_constraints_ and set the constraint as hard constraint in
  // lower_bounds and upper_bounds
  for (const auto& constraint_data : inequality_constraints_) {
    unsigned int free_deriv_counter = 0;
    const int derivative_hc = constraint_data->derivative;
    const double value_hc = constraint_data->value;

    for (size_t v = 0; v < vertices.size(); ++v) {
      for (int deriv = 0; deriv <= derivative_to_optimize; ++deriv) {
        if (!vertices[v].hasConstraint(deriv)) {
          if (deriv == derivative_hc) {
            for (size_t k = 0; k < dim; ++k) {
              unsigned int start_idx = k * n_free_constraints;
              lower_bounds->at(start_idx + free_deriv_counter) =
                  -std::abs(value_hc);
              upper_bounds->at(start_idx + free_deriv_counter) =
                  std::abs(value_hc);
            }
          }
          free_deriv_counter++;
        }
      }
    }
  }
}

template <int _N>
double PolynomialOptimizationNonLinear<_N>::computeTotalTrajectoryTime(
    const std::vector<double>& segment_times) {
  double total_time = 0;
  for (double t : segment_times) total_time += t;
  return total_time;
}

}  // namespace mav_trajectory_generation

namespace nlopt {

inline std::string returnValueToString(int return_value) {
  switch (return_value) {
    case nlopt::SUCCESS:
      return std::string("SUCCESS");
    case nlopt::FAILURE:
      return std::string("FAILURE");
    case nlopt::INVALID_ARGS:
      return std::string("INVALID_ARGS");
    case nlopt::OUT_OF_MEMORY:
      return std::string("OUT_OF_MEMORY");
    case nlopt::ROUNDOFF_LIMITED:
      return std::string("ROUNDOFF_LIMITED");
    case nlopt::FORCED_STOP:
      return std::string("FORCED_STOP");
    case nlopt::STOPVAL_REACHED:
      return std::string("STOPVAL_REACHED");
    case nlopt::FTOL_REACHED:
      return std::string("FTOL_REACHED");
    case nlopt::XTOL_REACHED:
      return std::string("XTOL_REACHED");
    case nlopt::MAXEVAL_REACHED:
      return std::string("MAXEVAL_REACHED");
    case nlopt::MAXTIME_REACHED:
      return std::string("MAXTIME_REACHED");
    default:
      return std::string("ERROR CODE UNKNOWN");
  }
}
}  // namespace nlopt

#endif  // MAV_TRAJECTORY_GENERATION_IMPL_POLYNOMIAL_OPTIMIZATION_NONLINEAR_IMPL_H_
