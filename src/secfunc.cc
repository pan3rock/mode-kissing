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

#include "secfunc.hpp"

#include <Eigen/Dense>
#include <complex>

using namespace Eigen;
using namespace std::complex_literals;
using Cmplx = std::complex<double>;

namespace {
void normalize(Eigen::ArrayXcd &e) {
  double vmax = e.abs().maxCoeff();
  if (vmax < 1.0e-40) {
    vmax = 1.0;
  }
  e /= vmax;
}
} // namespace

SecularFunction::SecularFunction(const Eigen::Ref<const Eigen::ArrayXXd> model,
                                 bool sh)
    : nl_(model.rows()), thk_(nl_ - 1), dns_(model.col(2)), vs_(model.col(3)),
      vp_(model.col(4)), sh_(sh), is_water_(vs_(0) == 0.0) {
  ArrayXd z = model.col(1);
  thk_ = z.tail(nl_ - 1) - z.head(nl_ - 1);

  if (is_water_) {
    vs_(0) = 1.0e-8;
    iwater_ = 1;
  } else {
    iwater_ = 0;
  }
}

double SecularFunction::evaluate(double f, double c) {
  ilvl_ = -1;
  if (sh_) {
    return evaluate_sh(f, c);
  } else {
    return evaluate_psv(f, c);
  }
}

double SecularFunction::evaluate(double f, double c, int ilvl) {
  ilvl_ = ilvl;
  if (sh_) {
    return evaluate_sh(f, c);
  } else {
    return evaluate_psv(f, c);
  }
}

double SecularFunction::evaluate_psv(double f, double c) {
  const double omega = 2.0 * M_PI * f;
  ArrayXd vs2 = vs_.pow(2);
  ArrayXd vp2 = vp_.pow(2);
  ArrayXd mu = dns_ * vs2;
  double c2 = pow(c, 2);
  ArrayXd r1 = vs2 / c2;
  double k = omega / c;

  ArrayXcd X(6);
  double tk = 2.0 - c2 / vs2(0);
  X << 2.0 * tk, -pow(tk, 2), 0.0, 0.0, -4.0, 2.0 * tk;
  X *= pow(mu(0), 2);
  normalize(X);

  int imax = nl_ - 1;
  if (ilvl_ > 0)
    imax = ilvl_;

  ArrayXcd Xnew(6);
  for (int i = 0; i < imax; ++i) {
    double r_si = dns_(i + 1) / dns_(i);
    double r_en = 2.0 * (r1(i) - r_si * r1(i + 1));

    double r_a = r_si + r_en;
    double r_a1 = r_a - 1.0;
    double r_b = 1.0 - r_en;
    double r_b1 = r_b - 1.0;

    double thk = thk_(i);
    Cmplx rk, Ca, Sa;
    if (c < vp_(i)) {
      double rk1 = sqrt(1.0 - c2 / vp2(i));
      double aa = k * rk1 * thk;
      rk = {rk1, 0.0};
      Ca = std::cosh(aa);
      Sa = std::sinh(aa);
    } else if (c > vp_(i)) {
      double rk1 = sqrt(c2 / vp2(i) - 1.0);
      double aa = k * rk1 * thk;
      rk = {0.0, rk1};
      Ca = std::cos(aa);
      Sa = std::sin(aa) * 1.0i;
    } else {
      rk = 0.0;
      Ca = 1.0;
      Sa = 0.0;
    }

    Cmplx sk, Cb, Sb;
    int scale = 0;
    if (c < vs_(i)) {
      double sk1 = sqrt(1.0 - c2 / vs2(i));
      sk = {sk1, 0.0};
      double bb = k * sk1 * thk;
      Cb = std::cosh(bb);
      Sb = std::sinh(bb);
      // Handle hyperbolic overflow
      if (k * sk.real() * thk > 80.0) {
        scale = 1;
        Cb /= Ca;
        Sb /= Ca;
      }
    } else if (c > vs_(i)) {
      double sk1 = sqrt(c2 / vs2(i) - 1.0);
      sk = {0.0, sk1};
      double bb = k * sk1 * thk;
      Cb = std::cos(bb);
      Sb = std::sin(bb) * 1.0i;
    } else {
      sk = 0.0;
      Cb = 1.0;
      Sb = 0.0;
    }

    if (i == 0 && is_water_) {
      Cb = 1.0;
      Sb = 0.0;
    }

    Cmplx pp1 = Cb * X(1) + sk * Sb * X(2);
    Cmplx pp2 = Cb * X(3) + sk * Sb * X(4);
    Cmplx pp3 = (1.0 / sk) * Sb * X(1) + Cb * X(2);
    Cmplx pp4 = (1.0 / sk) * Sb * X(3) + Cb * X(4);

    Cmplx qq1, qq2, qq3, qq4;
    if (scale == 1) {
      qq1 = pp1 - rk * pp2;
      qq2 = pp4 - pp3 / rk;
      qq3 = pp3 - rk * pp4;
      qq4 = pp2 - pp1 / rk;
    } else {
      qq1 = Ca * pp1 - rk * Sa * pp2;
      qq2 = Ca * pp4 - Sa * pp3 / rk;
      qq3 = Ca * pp3 - rk * Sa * pp4;
      qq4 = Ca * pp2 - Sa * pp1 / rk;
    }

    Cmplx yy1 = r_a * qq1;
    Cmplx yy2 = r_a1 * qq2;
    Cmplx zz1 = r_b1 * qq1;
    Cmplx zz2 = r_b * qq2;
    if (scale == 0) {
      yy1 += r_a1 * X(0);
      yy2 += r_a * X(0);
      zz1 += r_b * X(0);
      zz2 += r_b1 * X(0);
    }

    Xnew(0) = r_b1 * yy1 + r_b * yy2;
    Xnew(1) = r_a * yy1 + r_a1 * yy2;
    Xnew(2) = r_si * qq3;
    Xnew(3) = r_si * qq4;
    Xnew(4) = r_b1 * zz1 + r_b * zz2;
    Xnew(5) = Xnew(0);

    X = Xnew;
    normalize(X);
  }

  double dr = 1.0 - c2 / vp2(nl_ - 1);
  double ds = 1.0 - c2 / vs2(nl_ - 1);
  Cmplx rk = sqrt(dr + 0.0i);
  Cmplx sk = sqrt(ds + 0.0i);

  if (sk.real() < 0) {
    sk *= -1.0;
  }

  Cmplx dd = (X(1) + sk * X(2) - rk * (X(3) + sk * X(4)));
  return dd.real();
}

double SecularFunction::evaluate_sh(double f, double c) {
  double omega = 2.0 * M_PI * f;
  double wvno = omega / c;
  double beta1 = vs_(nl_ - 1);
  double rho1 = dns_(nl_ - 1);
  double xkb = omega / beta1;
  double wvnop = wvno + xkb;
  double wvnom = abs(wvno - xkb);
  double rb = sqrt(wvnop * wvnom);
  double e1 = rho1 * rb;
  double e2 = 1.0 / (beta1 * beta1);

  int mmin = iwater_;
  if (ilvl_ > 0)
    mmin = ilvl_ - 1;

  for (int m = nl_ - 2; m >= mmin; --m) {
    beta1 = vs_(m);
    rho1 = dns_(m);
    double xmu = rho1 * beta1 * beta1;
    xkb = omega / beta1;
    wvnop = wvno + xkb;
    wvnom = abs(wvno - xkb);
    rb = sqrt(wvnop * wvnom);
    double q = thk_(m) * rb;

    double sinq, y, z, cosq;
    if (wvno < xkb) {
      sinq = sin(q);
      y = sinq / rb;
      z = -rb * sinq;
      cosq = cos(q);
    } else if (wvno == xkb) {
      cosq = 1.0;
      y = thk_(m);
      z = 0.0;
    } else {
      double fac = 0.0;
      if (q < 16.0) {
        fac = exp(-2.0 * q);
      }
      cosq = (1.0 + fac) * 0.5;
      sinq = (1.0 - fac) * 0.5;
      y = sinq / rb;
      z = rb * sinq;
    }

    double e10 = e1 * cosq + e2 * xmu * z;
    double e20 = e1 * y / xmu + e2 * cosq;
    double norm = std::max(abs(e10), abs(e20));
    if (norm > 1.0e-40) {
      e1 = e10 / norm;
      e2 = e20 / norm;
    }
  }
  return e1;
}