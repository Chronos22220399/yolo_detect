#pragma once
#include <fstream>
#include <iostream>
// third_party
#include "../include/nlohmann/json.hpp"
// tools
#include "../include/JsonChecker.hpp"

struct SourcePaths {
  std::string imagePath;
  std::string videoPath;
  std::string cameraPath;
};

struct OutputPaths {
  std::string imagePath;
  std::string videoPath;
  std::string cameraPath;
};

struct Config {
  std::string modelPath;
  SourcePaths sourcePaths;
  OutputPaths outputPaths;
  std::vector<std::string> classNames;
};

// 实现 SourcesPath 的 from_json 函数
inline void from_json(const nlohmann::json &j, SourcePaths &src) {
  j.at("image").get_to(src.imagePath);
  j.at("video").get_to(src.videoPath);
  j.at("camera").get_to(src.cameraPath);
}

// 实现 OutputsPath 的 from_json 函数
inline void from_json(const nlohmann::json &j, OutputPaths &out) {
  j.at("image").get_to(out.imagePath);
  j.at("video").get_to(out.videoPath);
  j.at("camera").get_to(out.cameraPath);
}

// 实现 Config 的 from_json 函数
inline void from_json(const nlohmann::json &j, Config &config) {
  j.at("modelPath").get_to(config.modelPath);
  j.at("sourcePaths").get_to(config.sourcePaths);
  j.at("outputPaths").get_to(config.outputPaths);
  j.at("classNames").get_to(config.classNames);
}

class ConfigParser {
public:
  ConfigParser(const std::string &configFileName,
               const fileds_type &json_required_fields) {
    parseJson(configFileName, json_required_fields);
  }

  [[nodiscard]] Config getConfig() {
    if (_j.empty()) {
      std::cerr << "json 不能为空" << std::endl;
      exit(-1);
    }

    return _j.get<Config>();
  }

  void parseJson(const std::string &configFileName,
                 const fileds_type &json_required_fields) {
    std::ifstream ifs(configFileName);

    if (!ifs.is_open()) {
      std::cerr << "无法打开配置文件：" << configFileName << "\n";
      exit(-1);
    }

    try {
      ifs >> _j;

      JsonChecker::checkFields(_j, json_required_fields);
    } catch (const std::exception &e) {
      std::cerr << "获取配置出错：" << e.what() << "\n";
      exit(-1);
    }
  }

private:
  nlohmann::json _j;
};
