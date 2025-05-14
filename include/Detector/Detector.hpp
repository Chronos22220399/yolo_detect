#pragma once
#include "../include/ConfigParser.hpp"
#include "../include/Detection.h"
#include "../include/DetectionDrawer.hpp"
#include "../include/Model/DnnOnnxModel.hpp"
#include "../include/Model/OnnxModelOutputParser.hpp"
#include "../include/whiteBalance.hpp"
#include <opencv2/opencv.hpp>

inline cv::Mat resizeWithAspectRatio(
    const cv::Mat &src, const cv::Size &target_size,
    const cv::Scalar &padding_color = cv::Scalar(114, 114, 114)) {
  int src_w = src.cols;
  int src_h = src.rows;
  int target_w = target_size.width;
  int target_h = target_size.height;

  // 计算缩放比例
  float scale = std::min(target_w / (float)src_w, target_h / (float)src_h);
  int new_w = static_cast<int>(src_w * scale);
  int new_h = static_cast<int>(src_h * scale);

  // 缩放图像
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(new_w, new_h));

  // 计算边缘填充
  int top = (target_h - new_h) / 2;
  int bottom = target_h - new_h - top;
  int left = (target_w - new_w) / 2;
  int right = target_w - new_w - left;

  // 填充边缘
  cv::Mat output;
  cv::copyMakeBorder(resized, output, top, bottom, left, right,
                     cv::BORDER_CONSTANT, padding_color);
  return output;
}

template <size_t Size, size_t Rows> class Detector {
protected:
  std::string modelPath;
  SourcePaths sourcePaths;
  OutputPaths outputPaths;
  std::vector<std::string> classNames;
  bool useYUYV;
  std::unique_ptr<Model> model;
  std::unique_ptr<ModelOutputParser> parser;
  const int rows = Rows;
  const float confThreshold;

public:
  Detector(Config &&config, std::unique_ptr<Model> model)
      : modelPath(std::move(config.modelPath)),
        sourcePaths(std::move(config.sourcePaths)),
        outputPaths(std::move(config.outputPaths)),
        classNames(std::move(config.classNames)), useYUYV(config.useYUYV),
        confThreshold(config.confThreshold), model(std::move(model)) {
    this->parser = std::make_unique<DnnOnnxModelOutputParser>();
  }

  virtual void detect(bool showOutput = true, bool save = true) = 0;

  static bool showOutput(bool showOutput, const cv::Mat &frame,
                         int frameRate = 0) {
    if (showOutput) {
      cv::imshow("img", frame);
      if (cv::waitKey(frameRate) == 'q') {
        return true;
      }
    }
    return false;
  }

  void drawOnImage(const std::vector<Detection> &results, cv::Mat &img) {

    auto count = results.size();

    if (this->useYUYV)
      white_balance(img);

    for (auto result : results) {
      auto box = result.box;
      auto class_id = result.class_id;
      auto confidence = std::to_string(result.confidence);
      auto color = cv::Scalar{255, 255, 0};
      std::string label = classNames[class_id] + ": " + confidence;

      DetectionDrawer::draw(img, label, box, color, count);
    }
  }

  virtual ~Detector() = default;
};
