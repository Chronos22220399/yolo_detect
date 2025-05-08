#include <exception>
#include <iostream>
// third_party
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv4/opencv2/opencv.hpp>

int main() {
  try {
    // 1. 初始化 ONNX
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::Session session(env, "./models/best.onnx", Ort::SessionOptions{});

    // 2. 读取测试图像
    cv::Mat img = cv::imread("./resources/images/surprise-1.jpg");
    cv::Mat resized;
    cv::resize(img, resized, cv::Size{640, 640});

    // 3. 预处理: 转浮点 + 归一化
    resized.convertTo(resized, CV_32F, 1.0 / 255);

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    std::vector<float> input_tensor_values;
    for (int c = 0; c < 3; ++c) {
      input_tensor_values.insert(input_tensor_values.end(),
                                 (float *)channels[c].datastart,
                                 (float *)channels[c].dataend);
    }

    assert(input_tensor_values.size() == 3 * 640 * 640);

    // 4. 创建输入 Tensor
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    // 5. 运行推理
    const char *input_names[] = {"images"};
    const char *output_names[] = {"output0"};
    auto outputs = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1,
                               output_names, 1);

    // 6. 解析输出
    float *data = outputs[0].GetTensorMutableData<float>();

    auto type_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = type_info.GetShape();

    std::cout << "batch size: " << output_shape[0] << std::endl;
    std::cout << "预测框: " << output_shape[1] << std::endl;
    std::cout << "每个框的属性: " << output_shape[2] << std::endl;

    auto d = outputs[0].GetTensorMutableData<float>();
    std::cout << "每个框的置信度: " << output_shape[2] << std::endl;

  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
  }
  return 0;
}
