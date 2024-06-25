#ifndef SECFUNC_H_
#define SECFUNC_H_

#include <Eigen/Dense>

class SecularFunction {
public:
  SecularFunction(const Eigen::Ref<const Eigen::ArrayXXd> model, bool sh);
  double evaluate(double f, double c);
  double evaluate(double f, double c, int ilvl);

private:
  double evaluate_psv(double f, double c);
  double evaluate_sh(double f, double c);

  const int nl_;
  Eigen::ArrayXd thk_, dns_, vs_, vp_;
  const bool sh_;
  const bool is_water_;
  int iwater_;
  int ilvl_ = -1;
};

#endif