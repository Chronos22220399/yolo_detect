#pragma once
// third_party
#include <opencv2/opencv.hpp>
// tools
#include "../ConfigParser.hpp"
#include "../DetectionDrawer.hpp"
#include "../include/Detector/Detector.hpp"

class ImageDetector : public Detector {
public:
  ImageDetector(Config &&config) : Detector(std::move(config)) {}

  virtual void detect(float conf_threshold = 0.4, bool showOutput = true,
                      bool save = true) override {
    auto imagePath = sourcePaths.imagePath;
    img = cv::imread(imagePath);

    auto output = model->output(img, 1.0 / 255, cv::Size{640, 640}, true);
    auto data = (float *)output.data;

    auto results = parser->parse(classNames, data, rows, conf_threshold, img);

    drawOnImage(results, img);

    if (save) {
      cv::imwrite(outputPaths.imagePath, img);
    }
    this->showOutput(showOutput, img);
  }

private:
  cv::Mat img;
};
