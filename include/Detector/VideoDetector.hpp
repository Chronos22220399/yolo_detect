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

template <size_t Size, size_t Rows>
class VideoDetector : public Detector<Size, Rows> {
public:
  VideoDetector(Config &&config, int frame, int batchSize = 4)
      : Detector<Size, Rows>(std::move(config)), frameRate(frame),
        queue(batchSize * 2), batchSize(batchSize) {
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
        if (frame.channels() == 1) {
          // NV12 格式读取进来是 1 通道，需要专门转换
          cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR_NV12);
        } else if (frame.channels() == 2) {
          cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR_YUY2);
        } else if (frame.channels() == 3) {

        } else {
          std::cerr << "Unsupported channel count: " << frame.channels()
                    << std::endl;
          return;
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
    cap.open(this->sourcePaths.videoPath);
    cap.set(cv::CAP_PROP_FPS, frameRate);
    // if (this->useYUYV) {
    //   cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y',
    //   'V'));
    // }
    if (!cap.isOpened()) {
      std::cerr << "打开视频失败\n";
      exit(-1);
    }
  }

  void runInference() {
    cv::Mat output;
    std::vector<Detection> results;
    const cv::Size modelSize(Size, Size);
    while (running) {
      cv::Mat frame;
      if (!queue.pop(frame))
        continue;

      if (frame.channels() == 1)
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

      output =
          std::move(this->model->output(frame, 1.0 / 255, modelSize, true));
      float *data = (float *)output.data;

      results = std::move(this->parser->parse(
          this->classNames, data, this->rows, 0.4, frame, Size, Size));
      this->drawOnImage(results, frame);

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
  const int batchSize;
};
