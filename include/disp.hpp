#ifndef DISP_H_
#define DISP_H_

#include <Eigen/Dense>
#include <memory>

class SecularFunction;

class Dispersion {
public:
  Dispersion(const Eigen::Ref<const Eigen::ArrayXXd> model, bool sh);
  ~Dispersion();
  std::vector<double> search(double f, int num_mode,
                             const std::vector<double> &samples, int ilvl = -1);
  std::vector<double> search_pred(double f, int num_mode);
  double search_mode(double f, int mode);
  std::vector<double> get_samples(double f);
  double approx(double f, double c);

private:
  double evaluate_rayleigh_velocity();
  Eigen::ArrayXd secular_function(double freq, double c1,
                                  const Eigen::ArrayXd &dcs);

  const double ednn_ = 0.25;
  const int nfine_ = 1;
  const double ctol_ = 1.0e-6;

  const int nl_;
  Eigen::ArrayXd thk_, vs_, vp_;
  const bool sh_;
  const int itop_;
  std::unique_ptr<SecularFunction> sf_;
  double vs0_, vp0_, vs_min_, vs_max_, vs_hf_, rayv_;
  double dc_rough_;
  std::vector<int> ilvl_;
};

#endif