#pragma once
// third_party
#include <opencv2/opencv.hpp>
// tools
#include "../ConfigParser.hpp"
#include "../include/Detector/Detector.hpp"
#include "../DetectionDrawer.hpp"
#include "../OnnxModel.hpp"
#include "../OnnxModelOutputParser.hpp"

class ImageDetector : public Detector {
   virtual void detectAndSave(const std::vector<std::string> &className,
                            const Config &config, const std::string &srcName,
                            const std::string &outputName,
                            bool showOutput = true) override {
    auto modelPath = config.modelPath;
    auto imagePath = config.srcsPath + srcName;
    auto outputPath = config.outputsPath + outputName;

    auto img = cv::imread(imagePath);

    OnnxModel model(config.modelPath);

    auto output = model.output(img, 1.0 / 255, cv::Size{640, 640}, true);
    auto data = (float *)output.data;
    const int rows = 25200; // 80*80*3 +40*40*3 + 20*20*3
    const double conf_threshold = 0.4;

    OnnxModelOutputParser parser;
    auto results = parser.parse(className, data, rows, conf_threshold, img);

    for (auto result : results) {
      auto box = result.box;
      auto class_id = result.class_id;
      auto confidence = std::to_string(result.confidence);
      auto color = cv::Scalar{255, 255, 0};
      std::string label = className[class_id] + confidence;

      DetectionDrawer::draw(img, label, box, color);
    }

    cv::imwrite(outputPath, img);
    if (showOutput) {
      cv::imshow("img", img);
      cv::waitKey();
    }
  }
};
