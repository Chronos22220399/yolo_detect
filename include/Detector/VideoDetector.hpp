#pragma once
// stl
#include <iostream>
// third_party
#include <opencv2/opencv.hpp>
// tools
#include "../ConfigParser.hpp"
#include "../DetectionDrawer.hpp"
#include "../include/Detector/Detector.hpp"
#include "../OnnxModel.hpp"
#include "../OnnxModelOutputParser.hpp"


class VideoDetector : public Detector {
  virtual void detectAndSave(
                             const Config &config,
                             float conf_threshold, bool showOutput = true) override {
    auto modelPath = config.modelPath;
    auto videoPath = config.srcsPath;
    auto outputPath = config.outputsPath;
    auto &classNames = config.classNames;

    OnnxModel model(config.modelPath);

    try {
      cv::VideoCapture cap(videoPath);

      if (!cap.isOpened()) {
        std::cerr << "打开文件 < " << videoPath << " > 失败\n";
        exit(-1);
      }

      cv::Mat img;

      while (cap.read(img)) {
        if (img.empty()) {
          std::cerr << "读取到空图像帧，跳过\n";
          continue;
        }

        auto output = model.output(img, 1.0 / 255, cv::Size{640, 640}, true);
        auto data = (float *)output.data;
        const int rows = 25200;

        OnnxModelOutputParser parser;
        auto results = parser.parse(classNames, data, rows, conf_threshold, img);

        for (auto result : results) {
          auto box = result.box;
          auto class_id = result.class_id;
          auto confidence = std::to_string(result.confidence);
          auto color = cv::Scalar{255, 255, 0};
          std::string label = classNames[class_id] + ": " + confidence;

          DetectionDrawer::draw(img, label, box, color);
        }

        if (showOutput) {
          cv::imshow("img", img);
          if (cv::waitKey(1) == 'q')
            break;
        }
      }
      cap.release();
      cv::destroyAllWindows();
    } catch (const std::exception &e) {
      std::cerr << "出错: " << e.what() << std::endl;
    }
  }
};
