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
  virtual void detectAndSave(const std::vector<std::string> &className,
                             const Config &config, const std::string &srcName,
                             const std::string &outputName,
                             bool showOutput = true) override {
    auto modelPath = config.modelPath;
//    auto videoPath = config.srcsPath + srcName;
    auto outputPath = config.outputsPath + outputName;

    OnnxModel model(config.modelPath);

    try {
      cv::VideoCapture cap(0);
      cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
      cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);

      if (!cap.isOpened()) {
        std::cerr << "打开摄像头失败\n";
        exit(-1);
      }

      cv::Mat img;

      displayThread = std::move(std::thread(&CameraDetector::displayImpl, this));

      while (cap.read(img)) {
        if (img.empty()) {
          std::cerr << "读取到空图像帧，跳过\n";
          continue;
        }

        auto output = model.output(img, 1.0 / 255, cv::Size{640, 640}, true);
        auto data = (float *)output.data;
        const int rows = 25200;
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

        if (showOutput) {
          std::lock_guard<std::mutex> lock(queueMutex);
          if (displayQueue.size() < 10) { // 限制队列长度
            displayQueue.push(img.clone());
          }
          if (cv::waitKey(1) == 'q')
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

public:
  CameraDetector() = default;
  // 防止意外拷贝
  CameraDetector(const CameraDetector&) = delete;
  CameraDetector& operator=(const CameraDetector&) = delete;

  // 允许移动操作
  CameraDetector(CameraDetector&&) = delete;
  CameraDetector& operator=(CameraDetector&&) = delete;

  ~CameraDetector() override {
    stop_flag.store(true, std::memory_order_release);

    // 先通知队列
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      displayQueue = {}; // 清空队列
    }

    // 再等待线程结束
    if (displayThread.joinable()) {
      displayThread.join();
    }
  }
private:
  static void display(CameraDetector &instance)
  {
    instance.displayImpl();
  }

  void displayImpl()
  {
    while (!stop_flag) {
      if (!displayQueue.empty()) {
        cv::Mat frame;
        {
          std::lock_guard<std::mutex> lock(queueMutex);
          frame = displayQueue.front();
          displayQueue.pop();
        }
        cv::imshow("Result", frame);
        cv::waitKey(1);
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

  std::thread displayThread;
  std::atomic_bool stop_flag{false};
  std::mutex queueMutex;
  std::queue<cv::Mat> displayQueue;
};
