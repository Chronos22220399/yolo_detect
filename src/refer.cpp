#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

std::vector<std::string> load_list(std::string path) {
  std::vector<std::string> class_list;
  std::ifstream ifs(path); // 打开文件
  // if (!ifs.is_open()) { // 如果打开文件失败
  //     std::cerr << "Failed to open file\n";
  //
  // }
  std::string line;
  while (getline(ifs, line))
    class_list.push_back(line);

  ifs.close(); // 关闭文件

  return class_list;
}

const std::vector<cv::Scalar> colors = {
    cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 0)};

const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;

struct Detection {
  int class_id;
  float confidence;
  cv::Rect box;
};

cv::Mat letterbox(cv::Mat img, float &ratio, std::vector<float> &pad,
                  cv::Size shape) {
  /*
  img：输入图像
  ratio：预处理图像的缩放比
  pad：预处理图像的填充
  shape：预处理图像的尺寸

  */
  float rows = img.rows;              // 原始图像的高 1080
  float cols = img.cols;              // 原始图像的宽 1920
  float hratio = shape.height / rows; // 计算缩放比
  float wratio = shape.width / cols;
  ratio = MIN(hratio, wratio);
  int h = int(rows * ratio); // 1920*1080  -->  640*360
  int w = int(cols * ratio);
  int dw =
      (shape.width - w) / 2; // 为了使640*360 --> 640*640 需在图像上下各增加140
  int dh = (shape.height - h) / 2;
  cv::Size dsize = cv::Size(w, h);
  // cv::Mat im;
  if (rows != h || cols != w)
    cv::resize(img, img, dsize, 0, 0, cv::INTER_AREA);
  cv::Mat out;
  cv::copyMakeBorder(img, out, dh, dh, dw, dw, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  pad.push_back(dh);
  pad.push_back(dw);
  return out;
}
void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output,
            const std::vector<std::string> &className, cv::Size shapeSize) {
  /*
  image: 输入图像
  net: 读入的onnx文件
  output：本函数的输出
  className：标签的种类
  shapeSize：预处理图像的尺寸，即输入模型的图像尺寸

  */
  cv::Mat blob;
  float ratio = 0.0;
  std::vector<float> pad;
  cv::Mat input_image =
      letterbox(image, ratio, pad, shapeSize); // 对原始图像进行预处理
  cv::dnn::blobFromImage(input_image, blob, 1. / 255., shapeSize, cv::Scalar(),
                         true, false);
  net.setInput(blob);
  std::vector<cv::Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames()); // 模型推理

  float *data = (float *)outputs[0].data;
  const int rows = 25200; // 80*80*3 +40*40*3 + 20*20*3

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (int i = 0; i < rows; ++i) {
    float confidence = data[4]; // 检测框的置信度
    if (confidence >= 0.4) {
      float *classes_scores = data + 5;
      cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
      cv::Point class_id;
      double max_class_score;
      minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
      if (max_class_score > SCORE_THRESHOLD) {
        confidences.push_back(confidence);
        class_ids.push_back(class_id.x); // 检测框的类别
        // 图像后处理，将检测框从640*640的图像上还原到1920*1080的图像上
        float x = data[0]; // x,y表示检测框的中心点
        float y = data[1];
        float w = data[2]; // w，h表示检测框的宽和高
        float h = data[3];
        int left = int((x - 0.5 * w - pad[1]) / ratio);
        int top = int((y - 0.5 * h - pad[0]) / ratio);
        int width = int(w / ratio);
        int height = int(h / ratio);
        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
    data = data + className.size() + 5;
  }

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD,
                    nms_result); // 非极大值抑制
  for (int i = 0; i < nms_result.size(); i++) {
    int idx = nms_result[i];
    Detection result;
    result.class_id = class_ids[idx];
    result.confidence = confidences[idx];
    result.box = boxes[idx];
    output.push_back(result);
  }
}

int split(std::string str, std::string get[], char c) {
  int pos = 0;   // 指向字符分割开始的的位置
  int l_pos = 0; // 指向分割结束的位置
  int count = 0; // 计数，保留分割字符串的数量

  for (int i = 0; i < str.length(); i++) {
    if (i == str.length() - 1 &&
        str[i] != c) { // 若读到最后一个字符，将最后一个分割后的字符串填入get[]
      l_pos = str.length();
      get[count] = std::string(str, pos, l_pos - pos);
      pos = l_pos + 1;
      ++count;
    } else if (str[i] == c) { // 分割
      l_pos = i;
      get[count] = std::string(str, pos, l_pos - pos);
      pos = l_pos + 1;
      ++count;
    }
  }
  return count;
}

int _main() {
  static const std::vector<std::string> class_list = {
      "angry",   "contempt", "disgust", "fear",     "happy",
      "natural", "sad",      "sleepy",  "surprised"};
  cv::dnn::Net net;
  net = cv::dnn::readNet("./models/best.onnx");

  // std::vector<std::string> img_list = load_list("./resources/images");
  std::vector<std::string> img_list = {"./resources/images/surprise-1.jpg"};
  cv::Size shape = cv::Size(640, 640);
  int cnt = 1;
  for (std::string &x : img_list) {
    // std::string get[20];
    std::string path = x;
    cv::Mat frame = cv::imread(path);
    std::vector<Detection> output;
    detect(frame, net, output, class_list, shape);
    for (int i = 0; i < output.size(); ++i) {
      auto &detection = output[i];
      auto box = detection.box;
      auto classId = detection.class_id;
      auto confidence1 = std::to_string(detection.confidence);
      const auto color = colors[classId % colors.size()];
      cv::rectangle(frame, box, color, 3);
      cv::rectangle(frame, cv::Point(box.x, box.y - 20),
                    cv::Point(box.x + box.width, box.y), color, cv::FILLED);
      cv::putText(frame, class_list[classId].c_str(),
                  cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 0, 0));
      cv::putText(frame, confidence1, cv::Point(box.x + 40, box.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    // std::cout << "--\n";
    std::string save_path = "./save";
    save_path += std::to_string(cnt) + ".jpg";
    cv::imwrite(save_path, frame);
    cnt++;
  }

  return 0;
}
