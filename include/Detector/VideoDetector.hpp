#pragma once
// stl
#include <atomic>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
// third_party
#include <opencv2/opencv.hpp>
// tools
#include "../ConfigParser.hpp"
#include "../DetectionDrawer.hpp"
#include "../SafeQueue.hpp"
#include "../include/Detector/Detector.hpp"

class VideoDetector : public Detector {
public:
  VideoDetector(Config &&config, int frame)
      : Detector(std::move(config)), frameRate(frame), queue(3) {
    running = true;
    inferThread = std::thread(&VideoDetector::runInference, this);
  }

  virtual ~VideoDetector() {
    running = false;
    if (inferThread.joinable()) {
      inferThread.join();
    }
    cap.release();
    cv::destroyAllWindows();
  }

  virtual void detect(float conf_threshold = 0.4, bool showOutput = true,
                      bool save = true) override {
    try {
      setUpVideoCapture();

      while (running) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
          std::cerr << "读取帧失败，跳过\n";
          continue;
        }
        queue.push(frame);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    } catch (const std::exception &e) {
      std::cerr << "主线程出错: " << e.what() << "\n";
    }
  }

protected:
  virtual void setUpVideoCapture() {
    cap.open(sourcePaths.videoPath);
    cap.set(cv::CAP_PROP_FPS, frameRate);
    if (!cap.isOpened()) {
      std::cerr << "打开视频失败\n";
      exit(-1);
    }
  }

  void runInference() {
    cv::Mat output;
    std::vector<Detection> results;
    while (running) {
      cv::Mat frame;
      if (!queue.pop(frame))
        continue;

      if (frame.channels() == 1)
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

      output =
          std::move(model->output(frame, 1.0 / 255, cv::Size{640, 640}, true));
      float *data = (float *)output.data;

      results = std::move(parser->parse(classNames, data, rows, 0.4, frame));
      drawOnImage(results, frame);

      showOutput(true, frame, frameRate);
    }
  }

  void showOutput(bool showOutput, const cv::Mat &frame, int frameRate = 0) {
    if (showOutput) {
      cv::imshow("img", frame);
      if (cv::waitKey(frameRate) == 'q') {
        running = false;
      }
    }
  }

protected:
  cv::VideoCapture cap;
  SafeQueue<cv::Mat> queue;
  std::thread inferThread;
  std::atomic<bool> running;
  const int frameRate;
};

// class VideoDetector : public Detector {
// public:
//   VideoDetector(Config &&config) : Detector(std::move(config)), cap(0) {}
//
//   virtual void detect(float conf_threshold, bool showOutput = true,
//                       bool save = false) override {
//
//     try {
//       setUpVideoCapture();
//
//       while (cap.read(frame)) {
//         if (frame.empty()) {
//           std::cerr << "读取到空图像帧，跳过\n";
//           continue;
//         }
//
//         auto output = model->output(frame, 1.0 / 255, cv::Size{640, 640},
//         true); auto data = (float *)output.data; const int rows = 25200;
//
//         auto results =
//             parser->parse(classNames, data, rows, conf_threshold, frame);
//
//         drawOnImage(results, frame);
//
//         if (this->showOutput(showOutput, frame))
//           break;
//       }
//       cap.release();
//       cv::destroyAllWindows();
//     } catch (const std::exception &e) {
//       std::cerr << "出错: " << e.what() << std::endl;
//     }
//   }
//
// private:
//   void setUpVideoCapture() {
//     if (!cap.isOpened()) {
//       std::cerr << "打开摄像头失败\n";
//       exit(-1);
//     }
//   }
//
// private:
//   cv::VideoCapture cap;
//   cv::Mat frame;
// };
