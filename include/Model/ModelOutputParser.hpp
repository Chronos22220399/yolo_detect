#pragma once
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
// tools
#include "../include/Detection.h"

using class_list_type = std::vector<std::string>;

class ModelOutputParser {
protected:
  struct maxClassResult {
    float max_class_score;
    int class_id;
  };

public:
  virtual std::vector<Detection> parse(const class_list_type &class_list,
                                       float *data, int rows,
                                       float conf_threshold,
                                       const cv::Mat &srcImg) = 0;
};
