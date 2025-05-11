#pragma once
#include "../include/ConfigParser.hpp"

class Detector {
public:
  virtual void detectAndSave(
                             const Config &config,
                            float conf_threshold = 0.4, bool showOutput = true) = 0;

  virtual ~Detector() = default;
};
