cmake_minimum_required(VERSION 3.20)

set(TARGET Yolo)
set(ONNX_PATH ~/env/onnxruntime-linux-x64-gpu-1.15.1)

project(${TARGET})

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDES})
include_directories(${ONNX_PATH}/include)
link_directories(${ONNX_PATH}/lib)

add_library(${TARGET} SHARED Yolo.h Yolo.cpp)

target_link_libraries(${TARGET} ${OpenCV_LIBS} onnxruntime)