#include <exception>
#include <iostream>
// third_party
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
// tools
#include "../include/ConfigParser.hpp"
#include "../include/JsonChecker.hpp"
#include "../include/OnnxModel.hpp"
#include "../include/OnnxModelOutputParser.hpp"
#include "../include/DetectionDrawer.hpp"
#include "../include/Detector.hpp"

using namespace cv;

static const std::vector<std::string> className = {
    "angry",   "contempt", "disgust", "fear",     "happy",
    "natural", "sad",      "sleepy",  "surprised"};

static fileds_type json_required_fields = {"modelPath", "imagePath",
                                           "outputsPath"};



int main() {
  try {
    // 初始化
    ConfigParser configParser("/Users/wuming/Code/yolo_qt/config/config.json",
                              json_required_fields);

    auto config = configParser.getConfig();

    Detector::detectImageAndSave(className, config, "smile.jpg", "smile.jpg");

  } catch (const cv::Exception &e) {
    std::cout << "cv error: " << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
  };
  return 0;
}
