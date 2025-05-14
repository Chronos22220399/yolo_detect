#pragma once
#include "VideoDetector.hpp"

template <size_t Size, size_t Rows>
class CameraDetector : public VideoDetector<Size, Rows> {
public:
  CameraDetector(Config &&config, int frame)
      : VideoDetector<Size, Rows>(std::move(config), frame) {}

protected:
  void setUpVideoCapture() override {
    this->cap.open(0, cv::CAP_V4L2);
    this->cap.set(cv::CAP_PROP_FRAME_WIDTH, Size);
    this->cap.set(cv::CAP_PROP_FRAME_HEIGHT, Size);
    this->cap.set(cv::CAP_PROP_FPS, this->frameRate);
    // if (this->useYUYV) {
    //   this->cap.set(cv::CAP_PROP_FOURCC,
    //                 cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
    // }
    if (!this->cap.isOpened()) {
      std::cerr << "打开摄像头失败\n";
      exit(-1);
    }
  }
};
