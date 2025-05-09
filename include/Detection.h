#pragma once
#include <opencv2/opencv.hpp>

struct Detection {
  float confidence;
  int class_id;
  cv::Rect box;
};
