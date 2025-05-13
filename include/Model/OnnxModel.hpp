#pragma once
#include "./Model.hpp"
#include <opencv2/opencv.hpp>

class OnnxModel : public Model {
public:
  OnnxModel(const std::string &modelPath) : Model(modelPath) {
    try {
      net = cv::dnn::readNetFromONNX(modelPath);
    } catch (const cv::Exception &e) {
      std::cout << "cv error: " << e.what() << std::endl;
    }
  }

  ~OnnxModel() = default;

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

  cv::Mat output(std::vector<cv::Mat> &batch_imgs, double ratio,
                 const cv::Size &size, bool swapRB) override {
    // 使用OpenCV的blobFromImages直接生成批量blob
    // 参数说明：
    // - batch_imgs: 批量图像集合
    // - ratio: 缩放因子
    // - size: 目标尺寸（模型输入尺寸）
    // - swapRB: 是否交换红蓝通道
    cv::Mat blob = cv::dnn::blobFromImages(batch_imgs, ratio, size,
                                           cv::Scalar{}, swapRB, false, CV_32F);

    // 设置输入并前向传播
    net.setInput(blob);
    std::vector<cv::Mat> netOutput;
    net.forward(netOutput);

    return netOutput[0]; // 返回批量处理结果
  }

private:
  cv::dnn::Net net;
};
