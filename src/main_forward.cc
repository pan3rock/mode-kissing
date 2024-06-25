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
#include "utils.hpp"

#include <CLI11.hpp>
#include <Eigen/Dense>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <iostream>
#include <string>
#include <toml.hpp>
#include <vector>

using namespace Eigen;

int main(int argc, char const *argv[]) {
  CLI::App app{"Calculating dispersion curves with the delta-matrix method."};

  std::string file_config = "config.toml";
  app.add_option("-c,--config", file_config, "toml-type configure file");
  int mode_max = 0;
  app.add_option("-m,--mode", mode_max, "maximum mode up to");
  bool sh = false;
  app.add_flag("--sh", sh, "whether to compute Love waves");
  bool atten = false;
  app.add_flag("--atten", atten, "for viscoelastic model (inaccurate)");
  std::string file_model = "";
  app.add_option("--model", file_model, "filename of model");
  std::string file_out = "disp.txt";
  app.add_option("-o,--out", file_out, "filename of output");

  CLI11_PARSE(app, argc, argv);

  const auto config = toml::parse(file_config);
  auto dispersion = toml::find(config, "forward");
  if (file_model == "") {
    file_model = toml::find<std::string>(dispersion, "file_model");
  }

  auto model = loadtxt(file_model);

  Dispersion disp(model, sh);
  std::ofstream out(file_out);
  const auto fmin = toml::find<double>(dispersion, "fmin");
  const auto fmax = toml::find<double>(dispersion, "fmax");
  const auto nf = toml::find<int>(dispersion, "nf");
  ArrayXd freqs = ArrayXd::LinSpaced(nf, fmin, fmax);
  for (int i = 0; i < freqs.size(); ++i) {
    ArrayXd c = disp.search(freqs(i), mode_max + 1);
    for (int m = 0; m < c.rows(); ++m) {
      fmt::print(out, "{:15.5f}{:15.7f}{:15d}\n", freqs(i), c(m), m);
    }
  }
  out.close();

  return 0;
}