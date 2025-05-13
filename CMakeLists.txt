cmake_minimum_required(VERSION 3.15)
project(YOLO_QT_DETECTOR LANGUAGES CXX)

# 基础配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# 源文件配置
file(
  GLOB_RECURSE
  srcs
  CONFIGURE_DEPENDS
  include/*.h
  include/*.hpp
  src/*.c
  src/*.cpp
  src/*.cc)

# file(GLOB UI_FILES 
#      # ui/*.ui
# )
#
# file(GLOB RES_FILES
#     resources/*.qrc
#     resources/*.png
# )

# 项目组件配置
option(BUILD_WITH_OPENVINO "Enable OpenVINO backend" OFF)
option(BUILD_WITH_TENSORRT "Enable TensorRT backend" OFF)

# set(ONNXRUNTIME_ROOT /usr/include/onnxruntime)
# set(ONNXRUNTIME_INCLUDE_DIR "/usr/include/onnxruntime")
# set(ONNXRUNTIME_LIBS "/usr/lib/libonnxruntime.so")

# 依赖查找
# find_package(Qt5 COMPONENTS Core Gui Widgets Multimedia MultimediaWidgets Concurrent REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)


# 可执行目标
add_executable(${PROJECT_NAME}
    ${srcs}
    # ${UI_FILES}
    # ${RES_FILES}
)

# 包含目录
target_include_directories(${PROJECT_NAME}
                           PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include
    # ${ONNXRUNTIME_INCLUDE_DIR}  # 添加 ONNX Runtime 头文件
)

# 链接配置
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    # ${ONNXRUNTIME_LIBS}  # 直接链接库
)
