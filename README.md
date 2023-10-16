# Yolo
using ONNXRuntime to inference yolov5 / yolov8 ( detect / pose ) model in C++

## Dependencies
- *OpenCV* [REQUIRED]
- *onnxruntime* [REQUIRED]
- *cuda* [OPTIONAL]
- *cudnn* [OPTIONAL]

## Cmake
How to build your project with Yolo?

In order to avoid *Binary compatibility issues*,suggest you to compile dynamic link library by yourself.
```shell
git clone https://github.com/OoShawnoO/Yolo.git
```
modify ONNX_PATH in CMakeLists.txt with your onnxruntime dir path.
```shell
cd Yolo && mkdir build && cd build
cmake ..
```

Copy libYolo.so to your project lib.

Copy Yolo.h to your project include.

Write a CmakeLists.txt for your own project.
```cmake
set(ONNX_PATH your's onnxruntime dir path)

project(${TARGET})

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDES})
include_directories(${ONNX_PATH}/include)
link_directories(${ONNX_PATH}/lib)

add_executable(${TARGET} xxx.cpp Yolo.h)

target_link_libraries(${TARGET} ${OpenCV_LIBS} onnxruntime Yolo)
```


## Sample for yolov8 detect
```c++
/* using yolov8n.onnx for sample */
using namespace hzd;
// initialize yolov8 model
Yolo yolo("yolov8n.onnx",Yolo::DetectV8);
// read picture
cv::Mat mat = imread("xxx.jpg");
std::vector<Detections> detections;
// detect,it will spend a few milliseconds at first time
yolo.Detect(mat,detections);
// paint detections on mat
yolo.PaintDetections(mat,detections);
// show image
cv::imshow("frame",mat);
cv::waitKey(0);
```

## Sample for yolov8 pose
```c++
/* using yolov8n-pose.onnx for sample */
using namespace hzd;
// initialize yolov8-pose model
Yolo yolo("yolov8n-pose.onnx",Yolo::PoseV8);
// read picture
cv::Mat mat = imread("xxx.jpg");
std::vector<Person> persons;
// pose,it will spend a few milliseconds at first time
yolo.Pose(mat,detections);
// paint person body part on mat
yolo.PaintPersons(mat,persons);
// show image
cv::imshow("frame",mat);
cv::waitKey(0);
```