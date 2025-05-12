#!/bin/bash
# 基础构建

cmake -B build -G Ninja
cmake --build build --parallel 16

cd build
./YOLO_QT_DETECTOR
