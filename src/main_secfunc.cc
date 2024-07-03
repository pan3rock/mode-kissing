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
#include "timer.hpp"
#include "utils.hpp"

#include <CLI11.hpp>
#include <Eigen/Dense>
#include <fmt/format.h>
#include <highfive/H5Easy.hpp>
#include <string>
#include <toml.hpp>

using namespace Eigen;

int main(int argc, char const *argv[]) {
  CLI::App app{"Scanning secular function given a model and frequency."};

  double freq;
  app.add_option("freq", freq, "frequency")->required();
  std::string file_config = "config.toml";
  app.add_option("-c,--config", file_config, "toml-type configure file");
  bool sh = false;
  app.add_flag("--sh", sh, "whether to compute Love waves");
  int ilayer = -1;
  app.add_option("-i,--ilayer", ilayer, "index of layers");
  std::string file_model = "";
  app.add_option("--model", file_model, "filename of model");
  std::string file_out = "secfunc.h5";
  app.add_option("-o,--out", file_out, "filename of output");

  CLI11_PARSE(app, argc, argv);

  const auto config = toml::parse(file_config);
  auto conf_secfunc = toml::find(config, "secfunc");
  if (file_model == "") {
    file_model = toml::find<std::string>(conf_secfunc, "file_model");
  }
  const int nc = toml::find<int>(conf_secfunc, "nc");

  ArrayXXd model = loadtxt(file_model);
  SecularFunction sf(model, sh);

  double cmin = model.col(3).minCoeff() * 0.8;
  cmin = toml::find_or<double>(conf_secfunc, "cmin", cmin);
  double cmax = model(model.rows() - 1, 3);
  cmax = toml::find_or<double>(conf_secfunc, "cmax", cmax);
  ArrayXd c = ArrayXd::LinSpaced(nc, cmin, cmax);

  ArrayXd sfunc(nc);
  for (int i_c = 0; i_c < nc; ++i_c) {
    Timer::begin("sfunc");
    sfunc(i_c) = sf.evaluate(freq, c(i_c), ilayer);
    Timer::end("sfunc");
  }
  std::cout << Timer::summery() << std::endl;

  const int mode_max = 1000;
  Dispersion disp(model, sh);
  auto samples = disp.get_samples(freq);
  auto roots = disp.search_pred(freq, mode_max);
  ArrayXd N(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    N(i) = disp.approx(freq, samples[i]);
  }
  auto samples_pred = disp.predict_samples(freq, samples);

  H5Easy::File fout(file_out, H5Easy::File::Overwrite);
  H5Easy::dump(fout, "f", freq);
  H5Easy::dump(fout, "c", c);
  H5Easy::dump(fout, "sfunc", sfunc);
  H5Easy::dump(fout, "samples", samples);
  H5Easy::dump(fout, "samples_pred", samples_pred);
  H5Easy::dump(fout, "N", N);
  H5Easy::dump(fout, "roots", roots);

  return 0;
}
