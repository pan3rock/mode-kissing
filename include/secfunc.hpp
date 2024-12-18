#ifndef SECFUNC_H_
#define SECFUNC_H_

#include <Eigen/Dense>

class SecularFunction {
public:
  SecularFunction(const Eigen::Ref<const Eigen::ArrayXXd> model, bool sh);
  double evaluate(double f, double c);

private:
  double evaluate_sh(double f, double c);
  double buchen96(double f, double c);
  double dltar4(double freq, double c); // Dunkin 1965, same as CPS

  void var(double p, double q, double ra, double rb, double wvno, double xka,
           double xkb, double dpth, double &w, double &cosp, double &a0,
           double &cpcq, double &cpy, double &cpz, double &cqw, double &cqx,
           double &xy, double &xz, double &wy, double &wz);

  Eigen::ArrayXXd dnka(double wvno2, double gam, double gammk, double rho,
                       double a0, double cpcq, double cpy, double cpz,
                       double cqw, double cqx, double xy, double xz, double wy,
                       double wz);

  const int method_ = 0; // method = 0: Buchen 1996; method = 1: Dunkin 1965.

  int nl_;
  Eigen::ArrayXd thk_, dns_, vs_, vp_;
  const bool sh_;
  const bool is_water_;
  int iwater_;
};

#endif