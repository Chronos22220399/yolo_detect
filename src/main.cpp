#include <exception>
#include <iostream>
#include <memory>
// third_party
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
// tools
#include "../include/Detector/VideoDetector.hpp"
#include "../include/ConfigParser.hpp"
#include "../include/Detector/ImageDetector.hpp"
#include "../include/Detector/VideoDetector.hpp"
#include "../include/Detector/CameraDetector.hpp"
#include "../include/JsonChecker.hpp"

using namespace cv;

static const std::vector<std::string> classNames;

static fileds_type json_required_fields = {"modelPath", "srcsPath",
                                           "outputsPath", "classNames"};

int main(int argc, char *argv[]) {
  try {
    // 初始化
    ConfigParser configParser("../configs/yk_config.json", json_required_fields);

    auto config = configParser.getConfig();

    std::unique_ptr<Detector> detector;
    if (false)
    {
      detector = std::make_unique<VideoDetector>();
    } else
    {
      detector = std::make_unique<CameraDetector>();
    }
    // detector = std::make_unique<ImageDetector>();

    detector->detectAndSave(config, "777.jpg", "777.jpg", 0.4);
    // Detector::detectAndSave(configs, "smile.jpg", "smile.jpg");

  } catch (const cv::Exception &e) {
    std::cout << "cv error: " << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
  };
  return 0;
}
