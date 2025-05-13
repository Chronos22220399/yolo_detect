#pragma once
#include "VideoDetector.hpp"

class CameraDetector : public VideoDetector {
public:
  CameraDetector(Config &&config, int frame)
      : VideoDetector(std::move(config), frame) {}

protected:
  void setUpVideoCapture() override {
    cap.open(0, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 320);
    cap.set(cv::CAP_PROP_FPS, frameRate);
    if (!cap.isOpened()) {
      std::cerr << "打开摄像头失败\n";
      exit(-1);
    }
  }
};
