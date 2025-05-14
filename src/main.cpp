#include <exception>
#include <iostream>
#include <memory>
// third_party
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
// tools
#include "../include/ConfigParser.hpp"
#include "../include/Detector/CameraDetector.hpp"
#include "../include/Detector/ImageDetector.hpp"
#include "../include/Detector/VideoDetector.hpp"
#include "../include/JsonChecker.hpp"

using namespace cv;

static const std::vector<std::string> classNames;

static fileds_type json_required_fields = {"modelPath", "sourcePaths",
                                           "outputPaths", "classNames"};

enum class DemoMode { ImageDemo, VideoDemo, CameraDemo };

struct Options {
  std::optional<std::string> configPath;
  std::optional<DemoMode> demoMode;
  std::optional<int> frameRate;
};

Options handleOptions(int argc, char *argv[]) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--config" && i + 1 < argc) {
      options.configPath = argv[i + 1];
      i++;

    } else if (arg == "--source" && i + 1 < argc) {
      std::string param = argv[i + 1];
      if (param == "image")
        options.demoMode = DemoMode::ImageDemo;
      else if (param == "video")
        options.demoMode = DemoMode::VideoDemo;
      else if (param == "camera")
        options.demoMode = DemoMode::CameraDemo;
    }

    else if (arg == "--frame" && i + 1 < argc) {
      std::string param = argv[i + 1];
      if (!param.empty()) {
        options.frameRate = std::stoi(param);
      }
    }
  }

  return options;
}

void run(int argc, char *argv[]) {
  // cv::VideoCapture cap(0);
  // while (!cap.isOpened()) {
  //   std::cerr << "open error" << std::endl;
  //   exit(-1);
  // }
  //
  // cv::Mat frame;
  // while (cap.read(frame)) {
  //   cv::resize(frame, frame, cv::Size{640, 640});
  //   cv::imshow("frame", frame);
  //   if (cv::waitKey(30) == 'q') {
  //     return 1;
  //   }
  // }

  auto options = handleOptions(argc, argv);
  if (!options.configPath) {
    std::cerr << "Usage: " << argv[0] << " --config <path_to_config>"
              << std::endl;
    exit(-1);
  }

  if (!options.demoMode) {
    std::cout << "Default: " << argv[0] << " --source image" << std::endl;
    options.demoMode = DemoMode::ImageDemo;
  }

  if (!options.frameRate) {
    std::cout << "Default: " << argv[0] << " --frame 30" << std::endl;
    options.frameRate = 30;
  }

  try {
    // 初始化
    // const size_t Size = 640;
    // const size_t Rows = 25200;
    const size_t Size = 320;
    const size_t Rows = 6300;
    ConfigParser configParser(options.configPath.value(), json_required_fields);

    auto config = configParser.getConfig();

    std::unique_ptr<Detector<Size, Rows>> detector;
    if (options.demoMode == DemoMode::ImageDemo) {
      detector = std::make_unique<ImageDetector<Size, Rows>>(std::move(config));

    } else if (options.demoMode == DemoMode::VideoDemo) {
      detector = std::make_unique<VideoDetector<Size, Rows>>(
          std::move(config), options.frameRate.value());

    } else if (options.demoMode == DemoMode::CameraDemo) {
      detector = std::make_unique<CameraDetector<Size, Rows>>(
          std::move(config), options.frameRate.value());

    } else {
      std::cerr << "Error !!!" << std::endl;
      exit(-1);
    }

    detector->detect(true, options.frameRate.value());

  } catch (const cv::Exception &e) {
    std::cout << "cv error: " << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
  };
}

int main(int argc, char *argv[]) {
  run(argc, argv);

  return 0;
}
