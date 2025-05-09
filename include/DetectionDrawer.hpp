#pragma once
#include <opencv2/opencv.hpp>

class DetectionDrawer {
public:
  static void draw(cv::Mat &img, const std::string &label, const cv::Rect &box,
                   const cv::Scalar &color) {
    // 绘制检测框
    cv::rectangle(img, box, color);
    // 绘制标签背景框
    cv::rectangle(img, cv::Point(box.x, box.y - 20),
                  cv::Point(box.x + box.width, box.y), color, cv::FILLED);
    cv::putText(img, label, cv::Point(box.x, box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  }
};
