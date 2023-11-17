# Yolo
using ONNXRuntime to inference yolov5 / yolov8 ( detect / pose ) model in C++

使用ONNXRuntime 推理yolov5/yolov8的detect/pose模型。
## Dependencies
- *OpenCV* [REQUIRED]
- *onnxruntime* [REQUIRED]
- *cuda* [OPTIONAL]
- *cudnn* [OPTIONAL]

## Cmake
How to build your project with Yolo?

如何在自己的项目中使用Yolo?

In order to avoid *Binary compatibility issues*,suggest you to compile dynamic link library by yourself.

为了避免二进制兼容性问题，建议您自己在电脑上编译动态链接库。

```shell
git clone https://github.com/OoShawnoO/Yolo.git
```
modify ONNX_PATH in CMakeLists.txt with your onnxruntime dir path.

更新CMakeLists.txt中ONNX_PATH宏为您的onnxruntime文件夹路径
```shell
cd Yolo && mkdir build && cd build
cmake ..
```

Copy libYolo.so to your project lib.

将libYolo.so复制到您的项目lib文件夹下。

Copy Yolo.h to your project include.

将Yolo.h复制到您的项目include文件夹下。

Write a CmakeLists.txt for your own project.

为您的项目编写一个CMakeLists.txt文件。
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