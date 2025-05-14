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
  using clock = std::chrono::steady_clock;

public:
  VideoDetector(Config &&config, int frame, int batchSize = 4)
      : Detector<Size, Rows>(std::move(config)), frameRate(frame),
        queue(batchSize * 2), batchSize(batchSize) {
    running.store(true, std::memory_order_release);
    inferThread = std::thread(&VideoDetector::runInference, this);
  }

  virtual ~VideoDetector() {
    running.store(false, std::memory_order_release);
    if (inferThread.joinable()) {
      inferThread.join();
    }
    std::cout << "关闭视频\n";
    cap.release();
    cv::destroyAllWindows();
  }

  virtual void detect(bool showOutput = true, bool save = true) override {
    setUpVideoCapture();
    lastFrameTime = clock::now();
    cv::Size size{Size, Size};

    while (running) {
      auto now = clock::now();
      auto elapsed = now - lastFrameTime;

      // 动态计算需要等待的时间
      auto targetInterval = std::chrono::milliseconds(1000 / frameRate);
      if (elapsed < targetInterval) {
        auto waitTime = targetInterval - elapsed;
        std::this_thread::sleep_for(waitTime);
      }

      // 采集帧的逻辑保持不变
      cv::Mat frame;
      if (!cap.read(frame))
        break;

      frame = resizeWithAspectRatio(frame, size);

      queue.push(frame);
      lastFrameTime = clock::now(); // 更新时间戳

      // 动态调整队列容量
      if (queue.size() > batchSize * 4) {
        std::this_thread::sleep_for(targetInterval * 2);
      }
    }
  }

protected:
  virtual void setUpVideoCapture() {
    cap.open(this->sourcePaths.videoPath);
    if (!cap.isOpened()) {
      std::cerr << "打开视频失败: " << this->sourcePaths.videoPath << "\n";
      exit(-1);
    }

    // 打印实际视频参数
    double actualFPS = cap.get(cv::CAP_PROP_FPS);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "视频参数 - FPS:" << actualFPS << " 分辨率:" << width << "x"
              << height << " 总帧数:" << cap.get(cv::CAP_PROP_FRAME_COUNT)
              << "\n";

    if (this->useYUYV) {
      bool ret = cap.set(cv::CAP_PROP_FOURCC,
                         cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
      std::cout << "设置YUYV格式" << (ret ? "成功" : "失败") << "\n";
    }

    bool fpsRet = cap.set(cv::CAP_PROP_FPS, frameRate);
    std::cout << "设置FPS为" << frameRate << (fpsRet ? "成功" : "失败") << "\n";
  }

  void runInference() {
    cv::Mat output;
    std::vector<Detection> results;
    const cv::Size modelSize(Size, Size);
    while (running) {
      cv::Mat frame;
      if (!queue.try_pop(frame))
        continue;

      if (frame.channels() == 1)
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

      output =
          std::move(this->model->output(frame, 1.0 / 255, modelSize, true));
      float *data = (float *)output.data;

      results = std::move(this->parser->parse(this->classNames, data,
                                              this->rows, this->confThreshold,
                                              frame, Size, Size));
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
  std::chrono::time_point<clock> lastFrameTime;
};
