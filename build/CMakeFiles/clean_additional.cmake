# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "")
  file(REMOVE_RECURSE
  "CMakeFiles/YOLO_QT_DETECTOR_autogen.dir/AutogenUsed.txt"
  "CMakeFiles/YOLO_QT_DETECTOR_autogen.dir/ParseCache.txt"
  "YOLO_QT_DETECTOR_autogen"
  )
endif()
