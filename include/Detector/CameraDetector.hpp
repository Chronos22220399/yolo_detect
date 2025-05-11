#pragma once
// stl
#include <iostream>
#include <mutex>
#include <thread>
#include <queue>
// third_party
#include <opencv2/opencv.hpp>
// tools
#include "../ConfigParser.hpp"
#include "../DetectionDrawer.hpp"
#include "../include/Detector/Detector.hpp"
#include "../OnnxModel.hpp"
#include "../OnnxModelOutputParser.hpp"


class CameraDetector : public Detector {
  virtual void detectAndSave(const Config &config, const std::string &srcName,
                             const std::string &outputName,
                             float conf_threshold = 0.4, bool showOutput = true) override {
    auto modelPath = config.modelPath;
    auto classNames = config.classNames;

    OnnxModel model(config.modelPath);

    try {
      cv::VideoCapture cap(0);
      // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
      // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);

      if (!cap.isOpened()) {
        std::cerr << "打开摄像头失败\n";
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
          if (cv::waitKey(30) == 'q')
          {
            break;
          }
        }
      }

      cap.release();
      cv::destroyAllWindows();
    } catch (const std::exception &e) {
      std::cerr << "出错: " << e.what() << std::endl;
    }
  }
};
