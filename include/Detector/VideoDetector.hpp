#pragma once
// stl
#include <iostream>
// third_party
#include <opencv2/opencv.hpp>
// tools
#include "../ConfigParser.hpp"
#include "../DetectionDrawer.hpp"
#include "../include/Detector/Detector.hpp"

class VideoDetector : public Detector {
public:
  VideoDetector(Config &&config) : Detector(std::move(config)), cap(0) {}

  virtual void detect(float conf_threshold, bool showOutput = true,
                      int milsec = 30, bool save = false) override {

    try {
      auto cap = setUpVideoCapture();

      while (cap.read(frame)) {
        if (frame.empty()) {
          std::cerr << "读取到空图像帧，跳过\n";
          continue;
        }

        auto output = model->output(frame, 1.0 / 255, cv::Size{640, 640}, true);
        auto data = (float *)output.data;
        const int rows = 25200;

        auto results =
            parser->parse(classNames, data, rows, conf_threshold, frame);

        drawOnImage(results, frame);

        if (this->showOutput(showOutput, frame, milsec))
          break;
      }
      cap.release();
      cv::destroyAllWindows();
    } catch (const std::exception &e) {
      std::cerr << "出错: " << e.what() << std::endl;
    }
  }

private:
  cv::VideoCapture setUpVideoCapture() {
    if (!cap.isOpened()) {
      std::cerr << "打开摄像头失败\n";
      exit(-1);
    }
    return cap;
  }

private:
  cv::VideoCapture cap;
  cv::Mat frame;
};
