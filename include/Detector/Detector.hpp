#pragma once
#include "../include/ConfigParser.hpp"
#include "../include/Detection.h"
#include "../include/DetectionDrawer.hpp"
#include "../include/Model/OnnxModel.hpp"
#include "../include/Model/OnnxModelOutputParser.hpp"
#include <opencv2/opencv.hpp>

class Detector {
protected:
  std::string modelPath;
  SourcePaths sourcePaths;
  OutputPaths outputPaths;
  std::vector<std::string> classNames;
  std::unique_ptr<Model> model;
  std::unique_ptr<ModelOutputParser> parser;
  const int rows = 25200;

public:
  Detector(Config &&config)
      : modelPath(std::move(config.modelPath)),
        sourcePaths(std::move(config.sourcePaths)),
        outputPaths(std::move(config.outputPaths)),
        classNames(std::move(config.classNames)) {
    this->model = std::make_unique<OnnxModel>(modelPath);
    this->parser = std::make_unique<OnnxModelOutputParser>();
  }

  virtual void detect(float conf_threshold = 0.4, bool showOutput = true,
                      bool save = true) = 0;

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
    for (auto result : results) {
      auto box = result.box;
      auto class_id = result.class_id;
      auto confidence = std::to_string(result.confidence);
      auto color = cv::Scalar{255, 255, 0};
      std::string label = classNames[class_id] + ": " + confidence;

      DetectionDrawer::draw(img, label, box, color);
    }
  }

  virtual ~Detector() = default;
};
