#!/bin/bash
# 基础构建

cmake -B build
cmake --build build --parallel $(nproc)

./build/YOLO_QT_DETECTOR --config configs/yk_config_detection.json --source camera --frame 30
