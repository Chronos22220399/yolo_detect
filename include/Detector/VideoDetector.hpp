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
  VideoDetector(Config &&config, int frame, int batchSize = 4)
      : Detector(std::move(config)), frameRate(frame), queue(batchSize * 2),
        batchSize(batchSize) {
    running = true;
    inferThread = std::thread(&VideoDetector::runInferenceSingle, this);
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
    if (useYUYV) {
      cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
    }
    if (!cap.isOpened()) {
      std::cerr << "打开视频失败\n";
      exit(-1);
    }
  }

  void runInferenceSingle() {
    cv::Mat output;
    std::vector<Detection> results;
    const cv::Size modelSize(640, 640);
    while (running) {
      cv::Mat frame;
      if (!queue.pop(frame))
        continue;

      if (frame.channels() == 1)
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

      output = std::move(model->output(frame, 1.0 / 255, modelSize, true));
      float *data = (float *)output.data;

      results = std::move(parser->parse(classNames, data, rows, 0.4, frame));
      drawOnImage(results, frame);

      showOutput(true, frame, frameRate);
    }
  }

  void runInferenceBatch() {
    std::vector<cv::Mat> batch;
    const cv::Size modelSize(640, 640);

    while (running) {
      batch.clear();
      cv::Mat frame;

      while (batch.size() < batchSize && queue.try_pop(frame)) {
        if (frame.channels() == 1)
          cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        batch.push_back(frame.clone());
      }

      if (!batch.empty()) {
        cv::Mat output = model->output(batch, 1.0 / 255, modelSize, true);
        float *data = (float *)output.data;

        // 解析输出维度
        const int total_predictions = output.rows;
        if (total_predictions % batchSize != 0) {
          throw std::runtime_error("模型输出维度与批处理大小不匹配");
        }
        const int pred_per_image = total_predictions / batch.size();

        for (size_t i = 0; i < batch.size(); ++i) {
          float *imgData = output.ptr<float>(i * pred_per_image);

          // 解析单个图像结果
          auto results =
              parser->parse(classNames, imgData, pred_per_image, 0.4, batch[i]);

          // 绘制检测结果
          for (const auto &det : results) {
            DetectionDrawer::draw(batch[i], classNames[det.class_id], det.box,
                                  cv::Scalar(0, 255, 0));
          }

          // 显示结果
          cv::imshow("Detection", batch[i]);
          if (cv::waitKey(1) == 'q')
            running = false;
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
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
