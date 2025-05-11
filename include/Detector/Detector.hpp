#pragma once
#include "../include/ConfigParser.hpp"

class Detector {
public:
  virtual void detectAndSave(const std::vector<std::string> &className,
                             const Config &config, const std::string &srcName,
                             const std::string &outputName,
                             bool showOutput = true) = 0;

  virtual ~Detector() = default;
};
