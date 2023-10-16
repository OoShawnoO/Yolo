#ifndef YOLODETECTOR_YOLO_H
#define YOLODETECTOR_YOLO_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

namespace hzd {

    struct Detection {
        cv::Rect    box{};
        float       conf{};
        int         classId{};
    };

    struct BodyPart {
        cv::Point   position;
        float       vision;
    };

    struct Person : Detection {
        // 鼻子 / nose
        BodyPart    nose{};
        // 左眼 / left eye
        BodyPart    leftEye{};
        // 右眼 / right eye
        BodyPart    rightEye{};
        // 左耳 / left ear
        BodyPart    leftEar{};
        // 右耳 / right ear
        BodyPart    rightEar{};
        // 左肩 / left shoulder
        BodyPart    leftShoulder{};
        // 右肩 / right shoulder
        BodyPart    rightShoulder{};
        // 左肘 / left elbow
        BodyPart    leftElbow{};
        // 右肘 / right elbow
        BodyPart    rightElbow{};
        // 左手腕 / left wrist
        BodyPart    leftWrist{};
        // 右手腕 / right wrist
        BodyPart    rightWrist{};
        // 左臀 / left hip
        BodyPart    leftHip{};
        // 右臀 / right hip
        BodyPart    rightHip{};
        // 左膝 / left knee
        BodyPart    leftKnee{};
        // 右膝 / right knee
        BodyPart    rightKnee{};
        // 左踝关节 / left ankle
        BodyPart    leftAnkle{};
        // 右踝关节 / right ankle
        BodyPart    rightAnkle{};
    };

    class Yolo {
    public:
        using DeviceID = int;
        enum YoloVersion { DetectV8,PoseV8,DetectV5,PoseV5 };

        Yolo() = default;
        explicit Yolo(nullptr_t){};
        
        Yolo
        (
                const std::string&      weightFilePath,
                YoloVersion             version,
                cv::Size                size = {640,640},
                bool                    cuda = true,
                DeviceID                deviceId = 0,
                float                   confThreshold = 0.5,
                float                   iouThreshold = 0.4
        );

        ~Yolo();

        bool Detect(
                const cv::Mat&          frame,
                std::vector<Detection>& result
        );
        bool Pose(
                const cv::Mat&          frame,
                std::vector<Person>&    result
        );

        void PaintDetections(
                cv::Mat&                                    frame,
                const std::vector<Detection>&               detections,
                const std::unordered_map<int,cv::Scalar>&   colorScheme = {
                        {0,cv::Scalar{0,255,255}},
                        {1,cv::Scalar{230,230,250}},
                        {2,cv::Scalar{255,250,205}},
                        {3,cv::Scalar{84,255,159}},
                        {4,cv::Scalar{132,112,255}},
                        {5,cv::Scalar{0,206,209}},
                        {6,cv::Scalar{202,255,112}},
                        {7,cv::Scalar{255,236,139}},
                        {8,cv::Scalar{255,106,106}},
                        {9,cv::Scalar{255,165,0}},
                        {10,cv::Scalar{240,128,128}},
                        {11,cv::Scalar{255,0,255}},
                        {12,cv::Scalar{138,43,226}},
                        {13,cv::Scalar{255,174,185}},
                        {14,cv::Scalar{171,130,255}},
                        {15,cv::Scalar{255,225,255}},
                        {16,cv::Scalar{144,238,144}}
                }
        );

        void PaintPersons(
                cv::Mat&                                    frame,
                const std::vector<Person>&                  person,
                const std::unordered_map<int,cv::Scalar>&   colorScheme = {
                        {0,cv::Scalar{0,255,255}},
                        {1,cv::Scalar{230,230,250}},
                        {2,cv::Scalar{255,250,205}},
                        {3,cv::Scalar{84,255,159}},
                        {4,cv::Scalar{132,112,255}},
                        {5,cv::Scalar{0,206,209}},
                        {6,cv::Scalar{202,255,112}},
                        {7,cv::Scalar{255,236,139}},
                        {8,cv::Scalar{255,106,106}},
                        {9,cv::Scalar{255,165,0}},
                        {10,cv::Scalar{240,128,128}},
                        {11,cv::Scalar{255,0,255}},
                        {12,cv::Scalar{138,43,226}},
                        {13,cv::Scalar{255,174,185}},
                        {14,cv::Scalar{171,130,255}},
                        {15,cv::Scalar{255,225,255}},
                        {16,cv::Scalar{144,238,144}}
                }
        );

    private:
        static Ort::Env             env;
        YoloVersion                 version;
        Ort::SessionOptions         sessionOptions;
        Ort::Session                session = Ort::Session(nullptr);
        Ort::RunOptions             runOptions;

        cv::Size                    size{};
        int                         width{};
        int                         height{};
        int                         channel{};
        float                       confThreshold{};
        float                       iouThreshold{};

        std::vector<char*>          inputClasses;
        std::vector<char*>          outputClasses;
        std::vector<int64_t>        inputDims;
        std::vector<int64_t>        outputDims;
        std::vector<Ort::Value>     inputTensors;
        std::vector<Ort::Value>     outputTensors;
        std::vector<std::string>    outputNames;
        Ort::MemoryInfo             memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtInvalidAllocator,OrtMemType::OrtMemTypeDefault);

        bool preprocessing(const cv::Mat& frame,cv::Mat& temp);
        void letterbox(
                const cv::Mat& frame,
                cv::Mat& outImage,
                const cv::Scalar& color = cv::Scalar(114, 114, 114),
                bool auto_ = false,
                bool scaleFill = false,
                bool scaleUp = true,
                int stride = 32
        ) const;
        void scalecoords(const cv::Size& originalSize,cv::Rect& coords) const;
        void scalecoords(const cv::Size& originalSize,cv::Point& point) const;
    };

} // hzd

#endif
