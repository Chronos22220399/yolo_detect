#pragma once
#include <fstream>
#include <iostream>
// third_party
#include <nlohmann/json.hpp>
// tools
#include "../include/JsonChecker.hpp"

struct Config {
  std::string modelPath;
  std::string srcsPath;
  std::string outputsPath;
  std::vector<std::string> classNames;
};

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
    return Config{_j.at("modelPath").get<std::string>(),
                  _j.at("srcsPath").get<std::string>(),
                  _j.at("outputsPath").get<std::string>(),
      _j.at("classNames").get<std::vector<std::string>>()
    };
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
