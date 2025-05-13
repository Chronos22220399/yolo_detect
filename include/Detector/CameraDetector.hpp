#pragma once
// stl
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
// third_party
#include <opencv2/opencv.hpp>
// tools
#include "../ConfigParser.hpp"
#include "../DetectionDrawer.hpp"
#include "../include/Detector/Detector.hpp"

class CameraDetector : public Detector {
public:
  CameraDetector(Config &&config) : Detector(std::move(config)), cap(0) {}

  ~CameraDetector() {
    cap.release();
    cv::destroyAllWindows();
  }

  virtual void detect(float conf_threshold = 0.4, bool showOutput = true,
                      int milsec = 30, bool save = true) override {
    try {
      static cv::Mat output;
      static float *data;
      auto cap = setUpVideoCapture();

      while (cap.read(frame)) {
        if (frame.empty()) {
          std::cerr << "读取到空图像帧，跳过\n";
          continue;
        }

        if (frame.channels() == 1) {
          cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }

        output = model->output(frame, 1.0 / 255, cv::Size{640, 640}, true);
        data = (float *)output.data;

        const int rows = 25200;

        auto results =
            parser->parse(classNames, data, rows, conf_threshold, frame);

        drawOnImage(results, frame);

        if (this->showOutput(showOutput, frame, milsec))
          break;
      }

    } catch (const std::exception &e) {
      std::cerr << "出错: " << e.what() << std::endl;
    }
  }

private:
  cv::VideoCapture setUpVideoCapture() {
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
    // cap.set(cv::CAP_PROP_FPS, 30);

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

// class CameraDetector : public Detector {
//   virtual void detectAndSave(const Config &config,
//                              float conf_threshold = 0.4, bool showOutput =
//                              true) override {
//     auto modelPath = config.modelPath;
// <<<<<<< HEAD
//     //    auto videoPath = config.srcsPath + srcName;
//     auto outputPath = config.outputsPath + outputName;
// =======
//     auto classNames = config.classNames;
// >>>>>>> 9a843af7a9302f6be5409aaba639446d2e680dee
//
//     OnnxModel model(config.modelPath);
//
//     try {
//       cv::VideoCapture cap(0);
//       // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
//       // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
//
//       if (!cap.isOpened()) {
//         std::cerr << "打开摄像头失败\n";
//         exit(-1);
//       }
//
//       cv::Mat img;
//
// <<<<<<< HEAD
//       // displayThread =
//       //     std::move(std::thread(&CameraDetector::displayImpl, this));
//
// =======
// >>>>>>> 9a843af7a9302f6be5409aaba639446d2e680dee
//       while (cap.read(img)) {
//         if (img.empty()) {
//           std::cerr << "读取到空图像帧，跳过\n";
//           continue;
//         }
//
//         auto output = model.output(img, 1.0 / 255, cv::Size{640, 640}, true);
//         auto data = (float *)output.data;
//         const int rows = 25200;
//
//         OnnxModelOutputParser parser;
//         auto results = parser.parse(classNames, data, rows, conf_threshold,
//         img);
//
//         for (auto result : results) {
//           auto box = result.box;
//           auto class_id = result.class_id;
//           auto confidence = std::to_string(result.confidence);
//           auto color = cv::Scalar{255, 255, 0};
//           std::string label = classNames[class_id] + ": " + confidence;
//
//           DetectionDrawer::draw(img, label, box, color);
//         }
//
//         if (showOutput) {
// <<<<<<< HEAD
//           // std::lock_guard<std::mutex> lock(queueMutex);
//           // if (displayQueue.size() < 10) { // 限制队列长度
//           //   displayQueue.push(img.clone());
//           // }
//           cv::imshow("img", img);
//           if (cv::waitKey(1) == 'q') {
// =======
//           cv::imshow("img", img);
//           if (cv::waitKey(30) == 'q')
//           {
// >>>>>>> 9a843af7a9302f6be5409aaba639446d2e680dee
//             break;
//           }
//         }
//       }
//
//       cap.release();
//       cv::destroyAllWindows();
//     } catch (const std::exception &e) {
//       std::cerr << "出错: " << e.what() << std::endl;
//     }
//   }
// <<<<<<< HEAD
//
// public:
//   CameraDetector() = default;
//   // 防止意外拷贝
//   CameraDetector(const CameraDetector &) = delete;
//   CameraDetector &operator=(const CameraDetector &) = delete;
//
//   // 允许移动操作
//   CameraDetector(CameraDetector &&) = delete;
//   CameraDetector &operator=(CameraDetector &&) = delete;
//
//   ~CameraDetector() override {
//     // stop_flag.store(true, std::memory_order_release);
//     //
//     // // 先通知队列
//     // {
//     //   std::lock_guard<std::mutex> lock(queueMutex);
//     //   displayQueue = {}; // 清空队列
//     // }
//     //
//     // // 再等待线程结束
//     // if (displayThread.joinable()) {
//     //   displayThread.join();
//     // }
//   }
//
// private:
//   // static void display(CameraDetector &instance) { instance.displayImpl();
//   }
//
//   // void displayImpl() {
//   //   while (!stop_flag) {
//   //     if (!displayQueue.empty()) {
//   //       cv::Mat frame;
//   //       {
//   //         std::lock_guard<std::mutex> lock(queueMutex);
//   //         frame = displayQueue.front();
//   //         displayQueue.pop();
//   //       }
//   //       cv::imshow("Result", frame);
//   //       cv::waitKey(1);
//   //     } else {
//   //       std::this_thread::sleep_for(std::chrono::milliseconds(1));
//   //     }
//   //   }
//   // }
//
//   // std::thread displayThread;
//   // std::atomic_bool stop_flag{false};
//   // std::mutex queueMutex;
//   // std::queue<cv::Mat> displayQueue;
// =======
// >>>>>>> 9a843af7a9302f6be5409aaba639446d2e680dee
// };
