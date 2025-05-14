#include <opencv2/opencv.hpp>

inline cv::Mat white_balance(const cv::Mat &frame) {
  CV_Assert(frame.type() == CV_8UC3 || frame.type() == CV_32FC3);

  // 将输入图像转换为 float 类型以进行浮点运算（避免溢出）
  cv::Mat float_frame;
  frame.convertTo(float_frame, CV_32FC3);

  // 分离通道
  std::vector<cv::Mat> channels(3);
  cv::split(float_frame, channels);

  // 计算每个通道的平均值
  double avg_b = cv::mean(channels[0])[0];
  double avg_g = cv::mean(channels[1])[0];
  double avg_r = cv::mean(channels[2])[0];
  double avg = (avg_b + avg_g + avg_r) / 3.0;

  // 计算增益
  double b_gain = avg / avg_b;
  double g_gain = avg / avg_g;
  double r_gain = avg / avg_r;

  // 应用增益
  channels[0] *= b_gain;
  channels[1] *= g_gain;
  channels[2] *= r_gain;

  // 合并通道
  cv::merge(channels, float_frame);

  // 转换回原始类型（可选，视你的处理流程而定）
  cv::Mat balanced;
  float_frame.convertTo(balanced, frame.type());

  return balanced;
}
