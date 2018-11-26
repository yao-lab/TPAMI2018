#include "barrier.h"

#include <../lib/Eigen/Dense>
using namespace Eigen;

struct Range {
  int start;  // starting index
  int end;    // pass the end index

  Range(int s, int e) : start(s), end(e) {}
};

template<typename M>
void writeEigen(M A, const std::string & path){
  std::ofstream file(path);
  if (file.is_open()){
    file << A << endl;
  }
}

template<typename M>
M loadEigen(const std::string & path) {
    //typedef Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor> Mat;
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<float> values;
    int rows = 0;
    while (std::getline(indata, line)) {
      std::stringstream lineStream(line);
      float cell;
      while (!lineStream.eof()){
        lineStream >> cell;
        values.push_back(cell);
      }
      // while (std::getline(lineStream, cell, ' ')) {
      //     values.push_back(std::stod(cell));
      // }
      ++rows;
    }
    return Map<const Eigen::Matrix<typename M::Scalar, -1, -1, RowMajor>>(values.data(), rows, values.size()/rows);
}

