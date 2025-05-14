#pragma once
#include <opencv2/opencv.hpp>

class DetectionDrawer {
public:
  static void draw(cv::Mat &img, const std::string &label, const cv::Rect &box,
                   const cv::Scalar &color, int count = -1) {
    // 绘制检测框
    cv::rectangle(img, box, color);
    // 绘制标签背景框
    cv::rectangle(img, cv::Point(box.x, box.y - 20),
                  cv::Point(box.x + box.width, box.y), color, cv::FILLED);
    cv::putText(img, label, cv::Point(box.x, box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    // 若需要显示总计数信息（左上角）
    if (count >= 0) {
      std::string count_text = "Count: " + std::to_string(count);
      // 背景框
      cv::rectangle(img, cv::Point(5, 5), cv::Point(150, 30),
                    cv::Scalar(255, 255, 255), cv::FILLED);
      // 文字
      cv::putText(img, count_text, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                  0.6, cv::Scalar(0, 0, 0), 1);
    }
  }
};
