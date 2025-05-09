#pragma once
// third_party
#include <nlohmann/json.hpp>

using fileds_type = std::vector<std::string>;
class JsonChecker {
public:
  static bool checkFields(const nlohmann::json &j,
                          const fileds_type &required_fields) {
    for (auto &field : required_fields) {
      if (!j.contains(field)) {
        return false;
      }
    }
    return true;
  }
};
