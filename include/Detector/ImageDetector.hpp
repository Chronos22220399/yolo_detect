#pragma once
// third_party
#include <opencv2/opencv.hpp>
// tools
#include "../ConfigParser.hpp"
#include "../DetectionDrawer.hpp"
#include "../include/Detector/Detector.hpp"

template <size_t Size, size_t Rows>
class ImageDetector : public Detector<Size, Rows> {
public:
  ImageDetector(Config &&config, std::unique_ptr<Model> model)
      : Detector<Size, Rows>(std::move(config), std::move(model)) {}

  virtual void detect(bool showOutput = true, bool save = true) override {
    auto imagePath = this->sourcePaths.imagePath;
    img = cv::imread(imagePath);

    cv::resize(img, img, cv::Size{Size, Size});

    auto output =
        this->model->output(img, 1.0 / 255, cv::Size{Size, Size}, true);
    auto data = (float *)output.data;

    auto results = this->parser->parse(this->classNames, data, this->rows,
                                       this->confThreshold, img, Size, Size);

    this->drawOnImage(results, img);

    if (save) {
      cv::imwrite(this->outputPaths.imagePath, img);
    }
    this->showOutput(showOutput, img);
  }

private:
  cv::Mat img;
};
