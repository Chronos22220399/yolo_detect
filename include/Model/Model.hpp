#pragma once
#include <opencv2/opencv.hpp>

class Model {
public:
  Model(const std::string &modelPath) {}

  virtual cv::Mat output(const cv::Mat &input, double ratio,
                         const cv::Size &size, bool swapRB) = 0;

  virtual ~Model() = default;

private:
  cv::dnn::Net net;
};
