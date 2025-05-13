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
                      int milsec = 30, bool save = true) override {
    auto imagePath = sourcePaths.imagePath;
    img = cv::imread(imagePath);

    auto output = model->output(img, 1.0 / 255, cv::Size{640, 640}, true);
    auto data = (float *)output.data;
    const int rows = 25200; // 80*80*3 +40*40*3 + 20*20*3

    auto results = parser->parse(classNames, data, rows, conf_threshold, img);

    drawOnImage(results, img);

    if (save) {
      cv::imwrite(outputPaths.imagePath, img);
    }
    this->showOutput(showOutput, img, 0);
  }

private:
  cv::Mat img;
};
