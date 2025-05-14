#pragma once
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
// tools
#include "../include/Detection.h"
#include "./ModelOutputParser.hpp"

using class_list_type = std::vector<std::string>;

class DnnOnnxModelOutputParser : public ModelOutputParser {

public:
  std::vector<Detection> parse(const class_list_type &class_list, float *data,
                               int rows, float conf_threshold,
                               const cv::Mat &srcImg, size_t scale_row,
                               size_t scale_col) override {
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<cv::Rect> boxes;

    confidences.reserve(rows);
    class_ids.reserve(rows);
    boxes.reserve(rows);

    float scale_x = srcImg.cols / static_cast<float>(scale_col);
    float scale_y = srcImg.rows / static_cast<float>(scale_row);

    for (int i = 0; i < rows; ++i, data += class_list.size() + 5) {
      float conf = data[4];
      // 筛选信度较低的
      if (conf < conf_threshold)
        continue;

      // 获取当前图片的检测数据
      auto result =
          get_max_class_result(class_list, confidences, class_ids, boxes, data);

      if (result.max_class_score < 0.4)
        continue;

      float x = data[0];
      float y = data[1];
      float w = data[2];
      float h = data[3];

      int left_top_x = int(x - w / 2) * scale_x;
      int left_top_y = int(y - h / 2) * scale_y;
      int width = w * scale_x;
      int height = h * scale_y;

      confidences.push_back(result.max_class_score);
      class_ids.push_back(result.class_id);
      boxes.emplace_back(left_top_x, left_top_y, width, height);
    }

    return get_detection(confidences, boxes, class_ids);
  }

private:
  std::vector<Detection> get_detection(const std::vector<float> &confidences,
                                       const std::vector<cv::Rect> &boxes,
                                       const std::vector<int> &class_ids) {
    auto nms_result = get_nms_result(confidences, boxes, 0.2, 0.2);

    std::vector<Detection> detections;
    for (int i = 0; i < nms_result.size(); i++) {
      int idx = nms_result[i];
      Detection result;
      result.class_id = class_ids[idx];
      result.confidence = confidences[idx];
      result.box = boxes[idx];
      detections.push_back(result);
    }
    return detections;
  }

  std::vector<int> get_nms_result(const std::vector<float> &confidences,
                                  const std::vector<cv::Rect> &boxes,
                                  float score_threshold, float nms_threshold) {
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold, nms_threshold,
                      nms_result);
    return nms_result;
  }

  maxClassResult get_max_class_result(const class_list_type &class_list,
                                      std::vector<float> &confidences,
                                      std::vector<int> &class_ids,
                                      std::vector<cv::Rect> &boxes,
                                      float *data) {
    auto classes_scores = data + 5;
    float max_class_score = classes_scores[0];
    int max_class_id = 0;

    for (int i = 0; i < class_list.size(); ++i) {
      if (classes_scores[i] > max_class_score) {
        max_class_score = classes_scores[i];
        max_class_id = i;
      }
    }

    return maxClassResult{max_class_score, max_class_id};
  }
};

class Drawer {
public:
  static void draw_on_img(const cv::Mat &img, const cv::Rect &box,
                          const std::string &label, const cv::Scalar &color) {
    // 绘制检测框
    cv::rectangle(img, box, color);
    // 绘制标签背景框
    cv::rectangle(img, cv::Point(box.x, box.y - 20),
                  cv::Point(box.x + box.width, box.y), color, cv::FILLED);
    cv::putText(img, label, cv::Point(box.x, box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  }
};
