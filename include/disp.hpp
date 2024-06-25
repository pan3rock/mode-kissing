#ifndef DISP_H_
#define DISP_H_

#include <Eigen/Dense>
#include <memory>

class SecularFunction;

class Dispersion {
public:
  Dispersion(const Eigen::Ref<const Eigen::ArrayXXd> model, bool sh);
  ~Dispersion();
  Eigen::ArrayXd search(double f, int num_mode);
  double search_mode(double f, int mode);
  std::vector<double> get_samples(double f);

private:
  double evaluate_rayleigh_velocity();
  Eigen::ArrayXd secular_function(double freq, double c1,
                                  const Eigen::ArrayXd &dcs);
  double approx(double f, double c);

  const int nfine_ = 4;
  const double ctol_ = 1.0e-7;

  const int nl_;
  Eigen::ArrayXd thk_, vs_, vp_;
  const bool sh_;
  const int itop_;
  std::unique_ptr<SecularFunction> sf_;
  double vs0_, vp0_, vs_min_, vs_max_, vs_hf_, rayv_;
  double dc_rough_;
};

#endif