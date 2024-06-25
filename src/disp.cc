/*
 Copyright (c) 2024 Lei Pan

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#include "disp.hpp"
#include "secfunc.hpp"
#include "toms748.h"

#include <Eigen/Dense>
#include <fmt/format.h>
#include <limits>

using namespace Eigen;

// default value for toms748;
static const int defaultInterpolationsPerIteration = 2,
                 defaultMaximumIterations = 100;
static const double EPS = std::numeric_limits<double>::epsilon();
static const double defaultRelativeTolerance = EPS * 4;

namespace {
double calculate_newton_step(const double root, const double vp,
                             const double vs) {
  double p = pow(root, -1);
  double p2 = pow(root, -2);
  double vs_2 = pow(vs, -2);
  double vp_2 = pow(vp, -2);
  double xi = sqrt(p2 - vs_2);
  double eta = sqrt(p2 - vp_2);
  double func = pow(vs_2 - 2.0 * p2, 2) - 4.0 * xi * eta * p2;
  double deri = p2 * (8.0 * p * (vs_2 - 2 * p2) + 8.0 * p * xi * eta +
                      4.0 * p2 * p * (xi / eta + eta / xi));
  return func / deri;
}
} // namespace

Dispersion::Dispersion(const Eigen::Ref<const Eigen::ArrayXXd> model, bool sh)
    : nl_(model.rows()),
      thk_(model.col(1).tail(nl_ - 1) - model.col(1).head(nl_ - 1)),
      vs_(model.col(3)), vp_(model.col(4)), sh_(sh),
      itop_(model(0, 3) == 0 ? 1 : 0),
      sf_(std::make_unique<SecularFunction>(model, sh)) {
  vs0_ = vs_(0);
  vp0_ = vp_(0);
  vs_min_ = vs_.minCoeff();
  vs_max_ = vs_.maxCoeff();
  vs_hf_ = vs_(nl_ - 1);
  rayv_ = evaluate_rayleigh_velocity();
}

Dispersion::~Dispersion() = default;

double Dispersion::evaluate_rayleigh_velocity() {
  double root = 0.8 * vs0_;
  int count = 0;
  while (true) {
    double update_step = calculate_newton_step(root, vp0_, vs0_);
    root -= update_step;
    if (count > 10 || std::abs(update_step) < ctol_)
      break;
    ++count;
  }
  return root;
}

double Dispersion::approx(double f, double c) {
  double sum = 0.0;
  double c_2 = pow(c, -2);
  for (int i = itop_; i < nl_ - 1; ++i) {
    if (c > vs_(i)) {
      sum += sqrt(pow(vs_(i), -2) - c_2) * thk_(i);
    }
  }

  if (!sh_) {
    for (int i = 0; i < nl_ - 1; ++i) {
      if (c > vp_(i)) {
        sum += sqrt(pow(vp_(i), -2) - c_2) * thk_(i);
      }
    }
  }
  sum *= 2.0 * f;
  return sum;
}

std::vector<double> Dispersion::get_samples(double f) {
  std::vector<double> samples;
  // samples.push_back(vs_min_ - ctol_ * 100);
  // samples.push_back(vs_max_ + ctol_ * 100);

  double cmin;
  if (sh_) {
    cmin = vs_min_;
  } else {
    cmin = std::min(vs_min_, rayv_) * 0.9;
  }
  std::vector<double> pred;
  pred.push_back(cmin - ctol_ * 100);

  int nmax = static_cast<int>(std::floor(approx(f, vs_hf_))) + 1;
  for (int i = 0; i < nmax; ++i) {
    double n = static_cast<double>(i);
    auto func = [&](double c) -> double { return approx(f, c) - n; };
    if (func(cmin) * func(vs_max_) < 0) {
      double root = toms748(func, cmin, vs_max_, nullptr, ctol_ * 100,
                            defaultRelativeTolerance, defaultMaximumIterations,
                            defaultInterpolationsPerIteration);
      pred.push_back(root);
      cmin = root;
    }
  }
  pred.push_back(vs_max_ + ctol_ * 100);
  std::sort(pred.begin(), pred.end());
  for (int i = 0; i < nfine_; ++i) {
    int num = pred.size();
    for (int j = 0; j < num - 1; ++j) {
      double mid = (pred[j] + pred[j + 1]) / 2.0;
      pred.push_back(mid);
    }
    std::sort(pred.begin(), pred.end());
  }
  samples.insert(samples.end(), pred.begin(), pred.end());

  std::sort(samples.begin(), samples.end());

  if (!sh_) {
    samples.push_back(rayv_ - ctol_ * 100);
    samples.push_back(rayv_);
    samples.push_back(rayv_ + ctol_ * 100);
    std::sort(samples.begin(), samples.end());
  }

  return samples;
}

double Dispersion::search_mode(double f, int mode) {
  ArrayXd cs = search(f, mode + 1);
  if (cs.rows() < mode + 1) {
    return std::numeric_limits<double>::quiet_NaN();
  } else {
    return cs[mode];
  }
}

Eigen::ArrayXd Dispersion::search(double f, int num_mode) {
  std::function<double(double)> func = [&](double c) {
    return sf_->evaluate(f, c);
  };

  auto samples = get_samples(f);
  double c_prev = samples[0], c_curr;
  double f_prev = func(c_prev), f_curr;
  std::vector<double> find;
  int modes = 0;
  for (size_t i = 1; i < samples.size(); ++i) {
    c_curr = samples[i];
    f_curr = func(c_curr);
    if (std::isnan(f_curr))
      continue;
    if (f_curr * f_prev < 0.0) {
      double root = toms748(func, c_prev, c_curr, nullptr, ctol_,
                            defaultRelativeTolerance, defaultMaximumIterations,
                            defaultInterpolationsPerIteration);
      if (std::isnan(root)) {
        continue;
      }
      find.push_back(root);
      ++modes;
      if (modes >= num_mode)
        break;
    }

    c_prev = c_curr;
    f_prev = f_curr;
  }

  std::sort(find.begin(), find.end());
  ArrayXd ret = Map<ArrayXd, Unaligned>(find.data(), find.size());

  if (ret.size() > num_mode) {
    return ret.head(num_mode);
  } else {
    return ret;
  }
}