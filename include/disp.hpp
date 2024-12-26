#ifndef DISP_H_
#define DISP_H_

#include <Eigen/Dense>
#include <memory>

class SecularFunction;

std::vector<int> find_required_nl(const Eigen::ArrayXd &vs);

void append_samples_with_roots(std::vector<double> &samples,
                               const std::vector<double> roots);

void check_samples(std::vector<double> &samples, double vs_hf);

class Dispersion {
public:
  Dispersion(const Eigen::Ref<const Eigen::ArrayXXd> model, bool sh);
  ~Dispersion();
  std::vector<double> search(double f, int num_mode,
                             const std::vector<double> &samples);
  double search_mode(double f, int mode);
  std::vector<double> get_samples(double f);
  double approx(double f, double c);

private:
  double evaluate_rayleigh_velocity();
  Eigen::ArrayXd secular_function(double freq, double c1,
                                  const Eigen::ArrayXd &dcs);

  const double ednn_ = 0.50;
  const int nfine_ = 2;
  const double ctol_ = 1.0e-6;

  const int nl_;
  Eigen::ArrayXd thk_, vs_, vp_;
  const bool sh_;
  const int itop_;
  std::unique_ptr<SecularFunction> sf_;
  double vs0_, vp0_, vs_min_, vs_max_, vs_hf_, rayv_;
  double dc_rough_;
};

#endif