#include "utils.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace Eigen;
namespace fs = std::filesystem;

ArrayXXd loadtxt(const std::string &filename, const std::string &delim) {
  fs::path path(filename);
  if (!exists(path)) {
    throw std::runtime_error(
        fmt::format("file '{:s}' doesn't exist. ", filename));
  }

  // Read numbers from file into buffer.
  std::ifstream infile(filename);
  std::string line;
  std::string strnum;

  auto data = std::vector<std::vector<double>>();

  // clear first
  data.clear();

  // parse line by line
  while (getline(infile, line)) {
    if (line.empty()) {
      continue;
    }

    data.push_back(std::vector<double>());

    for (std::string::const_iterator i = line.begin(); i != line.end(); ++i) {
      // If i is not a delim, then append it to strnum
      if (delim.find(*i) == std::string::npos) {
        strnum += *i;
        if (i + 1 != line.end()) // If it's the last char, do not continue
          continue;
      }

      // if strnum is still empty, it means the previous char is also a
      // delim (several delims appear together). Ignore this char.
      if (strnum.empty())
        continue;

      // If we reach here, we got a number. Convert it to double.
      double number;

      std::istringstream(strnum) >> number;
      data.back().push_back(number);

      strnum.clear();
    }
  }

  ArrayXXd eArray(data.size(), data[0].size());
  for (size_t i = 0; i < data.size(); ++i)
    eArray.row(i) = Eigen::ArrayXd::Map(&data[i][0], data[0].size());
  return eArray;
}
