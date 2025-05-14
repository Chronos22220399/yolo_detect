#include "Model.hpp"
#include <onnxruntime/onnxruntime_cxx_api.h>

class OrtModel : public Model {
public:
  OrtModel(const std::string &modelPath)
      : Model(modelPath), env(ORT_LOGGING_LEVEL_WARNING, "test"),
        session(env, "./models/yk_emotion.onnx", Ort::SessionOptions{}) {}

  ~OrtModel() = default;

  cv::Mat output(const cv::Mat &input, double ratio, const cv::Size &size,
                 bool swapRB) override {

    cv::Mat img = cv::imread("./resources/yk/images/image.jpg");
    cv::Mat resized;
    cv::resize(img, resized, size);

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

    assert(input_tensor_values.size() == 3 * size.height * size.width);

    // 4. 创建输入 Tensor
    std::vector<int64_t> input_shape = {1, 3, size.width, size.height};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    // 5. 运行推理
    const char *input_names[] = {"images"};
    const char *output_names[] = {"output0"};
    auto outputs = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1,
                               output_names, 1);

    std::vector<int64_t> output_shape =
        outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int rows = output_shape[1];
    int cols = output_shape[2];

    // 6. 解析输出
    float *data = outputs[0].GetTensorMutableData<float>();

    cv::Mat result(rows, cols, CV_32F, data);
    return result;
  }

private:
  Ort::Env env;
  Ort::Session session;
};
