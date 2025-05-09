#include <exception>
#include <iostream>
// third_party
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
// tools
#include "../include/OnnxModel.hpp"

using namespace cv;

static const std::array<std::string, 9> className = {
    "angry",   "contempt", "disgust", "fear",     "happy",
    "natural", "sad",      "sleepy",  "surprised"};

struct Detection {
  float confidence;
  int class_id;
  cv::Rect box;
};

using class_list_type = decltype(className);

class OnnxModelOutputParser {
  struct minMaxResult {
    double max_class_score;
    int class_id;
  };

public:
  std::vector<Detection> parse(const class_list_type &class_list, float *data,
                               int rows, double conf_threshold) {
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i, data = data + class_list.size() + 5) {
      float conf = data[4];
      // 筛选信度较低的
      if (conf < conf_threshold)
        continue;

      // 获取当前图片的检测数据
      auto result =
          get_min_max_loc(class_list, confidences, class_ids, boxes, data);

      if (result.max_class_score < 0.4)
        continue;

      float x = data[0];
      float y = data[1];
      float w = data[2];
      float h = data[3];

      int left_top_x = int(x - w / 2);
      int left_top_y = int(y - h / 2);
      int width = w;
      int height = h;

      confidences.push_back(result.max_class_score);
      class_ids.push_back(result.class_id);
      boxes.emplace_back(left_top_x, left_top_y, width, height);

    }

    return get_detection(confidences, boxes, class_ids);
  }

private:
  std::vector<Detection> get_detection(const std::vector<float> &confidences,
                                       const std::vector<cv::Rect> &boxes,
                                       const std::vector<int> &class_ids) {
    auto nms_result = get_nms_result(confidences, boxes, 0.2, 0.2);

    std::vector<Detection> detections;
    for (int i = 0; i < nms_result.size(); i++) {
      int idx = nms_result[i];
      Detection result;
      result.class_id = class_ids[idx];
      result.confidence = confidences[idx];
      result.box = boxes[idx];
      detections.push_back(result);
    }
    return detections;
  }

  std::vector<int> get_nms_result(const std::vector<float> &confidences,
                                  const std::vector<cv::Rect> &boxes,
                                  float score_threshold, float nms_threshold) {
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold, nms_threshold,
                      nms_result);
    return nms_result;
  }

  minMaxResult get_min_max_loc(const class_list_type &class_list,
                               std::vector<float> &confidences,
                               std::vector<int> &class_ids,
                               std::vector<cv::Rect> &boxes, float *data) {
    auto classes_scores = data + 5;
    cv::Mat scores(1, class_list.size(), CV_32FC1, classes_scores);
    // 获取最可能的分类
    cv::Point class_id;
    double max_class_score;
    cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

    return {max_class_score, class_id.x};
  }
};

class Drawer {
public:
  static void draw_on_img(const cv::Mat &img, const cv::Rect &box,
                          const std::string &label, const cv::Scalar &color) {
    // 绘制检测框
    cv::rectangle(img, box, color);
    // 绘制标签背景框
    cv::rectangle(img, cv::Point(box.x, box.y - 20),
                  cv::Point(box.x + box.width, box.y), color, cv::FILLED);
    cv::putText(img, label, cv::Point(box.x, box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  }
};

int main() {
  try {
    // 初始化
    std::string path = "/home/Ess/Code/ClassDesign/Example/yolo_qt/";
    OnnxModel model(path + "models/best.onnx");
    auto img = imread(path + "resources/images/A276_jpg.rf.b291b0c810848f2866546d69a48f0504.jpg");
    auto output = model.output(img, 1.0 / 255, cv::Size{640, 640}, true);

    auto data = (float *)output.data;

    const int rows = 25200; // 80*80*3 +40*40*3 + 20*20*3
    const double conf_threshold = 0.4;

    OnnxModelOutputParser parser;
    auto results = parser.parse(className, data, rows, conf_threshold);

    for (auto result : results) {
      auto box = result.box;
      auto class_id = result.class_id;
      auto confidence = std::to_string(result.confidence);

      auto color = cv::Scalar{255, 255, 0};
      std::string label = className[class_id] + confidence;
      // 绘制检测框
      cv::rectangle(img, box, color);
      // 绘制标签背景框
      cv::rectangle(img, cv::Point(box.x, box.y - 20),
                    cv::Point(box.x + box.width, box.y), color, cv::FILLED);
      cv::putText(img, label, cv::Point(box.x, box.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("img", img);
    cv::imwrite("img.jpg", img);
    cv::waitKey();

  } catch (const cv::Exception &e) {
    std::cout << "cv error: " << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
  };
  return 0;
}
