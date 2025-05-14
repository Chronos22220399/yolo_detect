#pragma once
#include "./Model.hpp"
#include <opencv2/opencv.hpp>

class DnnOnnxModel : public Model {
public:
  DnnOnnxModel(const std::string &modelPath) : Model(modelPath) {
    try {
      net = cv::dnn::readNetFromONNX(modelPath);
    } catch (const cv::Exception &e) {
      std::cout << "cv error: " << e.what() << std::endl;
    }
  }

  ~DnnOnnxModel() = default;

  cv::Mat output(const cv::Mat &batchImg, double ratio, const cv::Size &size,
                 bool swapRB) override {
    cv::Mat resizedFrame;
    cv::resize(batchImg, resizedFrame, size);
    cv::Mat blob =
        cv::dnn::blobFromImage(batchImg, ratio, size, cv::Scalar{}, swapRB);

    // 输入
    net.setInput(blob);
    // 输出
    std::vector<cv::Mat> netOutput;
    net.forward(netOutput);

    cv::Mat output = netOutput[0];
    if (output.dims == 4) {
      int num_detections = output.size[2];
      int det_length = output.size[3];
      // 修复1：使用正确整数参数，计算总行数 = 批次大小 * 检测数量
      int total_rows = batchImg.rows * num_detections;
      // 修复2：移除重复的return语句
      return output.reshape(1, total_rows);
    }
    return output;
  }

private:
  cv::dnn::Net net;
};
