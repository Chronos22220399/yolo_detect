#include <opencv2/opencv.hpp>

class OnnxModel {
public:
  OnnxModel(const std::string &modelPath) {
    try {
      net = cv::dnn::readNetFromONNX(modelPath);
    } catch (const cv::Exception &e) {
      std::cout << "cv error: " << e.what() << std::endl;
    }
  }

  cv::Mat output(const cv::Mat &img, double ratio, const cv::Size &size,
                 bool swapRB) {
    cv::Mat resized_img;
    cv::resize(img, resized_img, size);
    cv::Mat blob =
        cv::dnn::blobFromImage(img, ratio, size, cv::Scalar{}, swapRB);

    // 输入
    net.setInput(blob);
    // 输出
    std::vector<cv::Mat> netOutput;
    net.forward(netOutput);

    return netOutput[0];
  }

private:
  cv::dnn::Net net;
};
