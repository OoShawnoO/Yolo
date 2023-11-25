#include <iostream>
#include <utility>
#include <opencv2/dnn/dnn.hpp>
#include "Yolo.h"

namespace hzd {
#define POSE_PREPARE(pose) do { \
    pose.position.x = (int)kps[pose_x_offset + k * 3 * anchorsCount + i]; \
    pose.position.y = (int)kps[pose_y_offset + k * 3 * anchorsCount + i]; \
    pose.vision = kps[pose_value_offset + k * 3 * anchorsCount + i];      \
    scaleCoords({frame.cols,frame.rows},pose.position);                   \
}while(0)

#define POSE_PAINT(pose,mat,index) do {                                                 \
    if(pose.vision > confThreshold) {                                                   \
        cv::circle(mat,pose.position,5,colorScheme.at(index),-1,cv::LINE_AA);           \
    }\
}while(0)

#define POSE_LINE(pose1,pose2,mat) do {                                                     \
    if(pose1.vision > confThreshold && pose2.vision > confThreshold){                       \
        cv::line(mat,pose1.position,pose2.position,cv::Scalar{72,118,255},3,cv::LINE_AA);   \
    }                                                                                       \
}while(0)

    Ort::Env Yolo::env;

    Yolo::Yolo(
            const std::string       &weightFilePath,
            Yolo::YoloVersion       _version,
            cv::Size                _size,
            bool                    cuda,
            DeviceID                deviceId,
            float                   _confThreshold,
            float                   _iouThreshold
    )
    :version(_version),size(std::move(_size)),confThreshold(_confThreshold),iouThreshold(_iouThreshold)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        if(cuda && OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions,deviceId)) {
            std::cerr << "[ WARN ] using cuda failed!" << std::endl;
        }
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session{env,weightFilePath.c_str(),sessionOptions};

        inputClasses.resize(session.GetInputCount());
        for(int i=0;i<inputClasses.size();i++){
            inputClasses[i] = new char[strlen(session.GetInputNameAllocated(i,allocator).get()) + 1];
            strcpy(inputClasses[i],session.GetInputNameAllocated(i,allocator).get());
        }
        outputClasses.resize(session.GetOutputCount());
        for(int i=0;i<outputClasses.size();i++){
            outputClasses[i] = new char[strlen(session.GetOutputNameAllocated(i,allocator).get()) + 1];
            strcpy(outputClasses[i],session.GetOutputNameAllocated(i,allocator).get());
        }

        width = (int)session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[2];
        height = (int)session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[3];
        channel = (int)session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1];

        inputDims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        outputDims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        std::string names = session.GetModelMetadata().LookupCustomMetadataMapAllocated("names",allocator).get();
        auto pos = names.find('\'');
        while(pos != std::string::npos) {
            auto newPos = names.find('\'',pos + 1);
            outputNames.emplace_back(names.substr(pos + 1,newPos - pos -1));
            pos = names.find('\'',newPos + 1);
        }
    }

    bool Yolo::preprocessing(const cv::Mat &frame,cv::Mat& temp) {
        if(frame.empty()) return false;

        letterbox(frame,temp);

        cv::cvtColor(temp,temp,cv::COLOR_BGR2RGB);
        temp.convertTo(temp,CV_32F,1.0f / 255.0f);

        cv::Mat channels[3];
        cv::split(temp,channels);

        temp = cv::Mat(1,size.area()*channel,CV_32F);

        for(int c = 0;c < 3;++c){
            channels[c]
            .reshape(1,1)
            .copyTo(temp.colRange(c*size.area(),(c+1)*size.area()));
        }

        inputTensors.emplace_back(Ort::Value::CreateTensor<float>(
                memoryInfo,
                (float*)temp.data,
                width * height *channel,
                inputDims.data(),
                inputDims.size()
        ));

        return true;
    }

    void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                          float& bestConf, int& bestClassId)
    {
        bestClassId = 5;
        bestConf = 0;

        for (int i = 5; i < numClasses + 5; i++)
        {
            if (it[i] > bestConf)
            {
                bestConf = it[i];
                bestClassId = i - 5;
            }
        }

    }

    bool Yolo::Detect(
            const cv::Mat           &frame,
            std::vector<Detection>  &result
    ) {
        if(version != DetectV5 && version != DetectV8) {
            std::cerr << "[ ERROR ] the weight file isn't yolo detect" << std::endl;
            exit(-1);
        }

        cv::Mat temp;
        inputTensors.clear();
        outputTensors.clear();
        if(!preprocessing(frame,temp)) return false;

        try{
            outputTensors = session.Run(Ort::RunOptions{nullptr},
                                        inputClasses.data(),
                                        inputTensors.data(),
                                        inputTensors.size(),
                                        outputClasses.data(),
                                        outputClasses.size());
        }catch(...){
            std::cout << "session Run failed." << std::endl;
            exit(-1);
        }

        auto* raw = outputTensors[0].GetTensorMutableData<float>();
        std::vector<cv::Rect>   boxes;
        std::vector<float>      confs;
        std::vector<int>        classIds;

        switch(version) {
            case DetectV5 : {
                size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
                std::vector<float> output(raw, raw + count);
                int numClasses = (int)outputDims[2] - 5;
                int elementsInBatch = (int)(outputDims[1] * outputDims[2]);

                for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputDims[2])
                {
                    float clsConf = it[4];

                    if (clsConf > confThreshold)
                    {
                        int centerX = (int) (it[0]);
                        int centerY = (int) (it[1]);
                        int _width = (int) (it[2]);
                        int _height = (int) (it[3]);
                        int left = centerX - _width / 2;
                        int top = centerY - _height / 2;

                        float objConf;
                        int classId;
                        getBestClassInfo(it, numClasses, objConf, classId);

                        float confidence = clsConf * objConf;

                        cv::Rect rect{left,top,_width,_height};
                        scaleCoords({frame.cols, frame.rows}, rect);

                        boxes.emplace_back(rect);
                        confs.emplace_back(confidence);
                        classIds.emplace_back(classId);
                    }
                }
                std::vector<int> indices;
                cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
                result.clear();
                for (int idx : indices)
                {
                    result.emplace_back(Detection{boxes[idx],confs[idx],classIds[idx]});
                }

                return true;
            }
            case DetectV8 : {
                int strideNum       = (int)outputDims[2];
                int signalResultNum = (int)outputDims[1];
                cv::Mat rowData(signalResultNum, strideNum, CV_32F, raw);
                rowData = rowData.t();
                auto* data = (float*)rowData.data;
                for (int i = 0; i < strideNum; ++i) {
                    float *classesScores = data + 4;
                    cv::Mat scores(1, signalResultNum-4, CV_32FC1, classesScores);
                    cv::Point class_id;
                    double maxClassScore;
                    cv::minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &class_id);
                    if (maxClassScore > confThreshold) {
                        confs.push_back((float)maxClassScore);
                        classIds.push_back(class_id.x);

                        float x = data[0];
                        float y = data[1];
                        float w = data[2];
                        float h = data[3];

                        int left = int(x-0.5 * w);
                        int top = int(y-0.5 * h);
                        int _width = (int)w;
                        int _height = (int)h;


                        cv::Rect rect{left,top,_width,_height};
                        scaleCoords({frame.cols, frame.rows}, rect);
                        boxes.emplace_back(rect);
                    }
                    data += signalResultNum;
                }
                std::vector<int> nmsResult;
                cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, nmsResult);
                std::vector<Detection> ret;
                result.clear();
                for (int idx : nmsResult) {
                    result.emplace_back(Detection{boxes[idx],confs[idx],classIds[idx]});
                }
                return true;
            }
            default : {
                std::cerr << "[ ERROR ] the weight file isn't yolo detect" << std::endl;
                exit(-1);
            }
        }
    }

    bool Yolo::Pose(
            const cv::Mat           &frame,
            std::vector<Person>     &result
    ) {
        if(version != PoseV5 && version != PoseV8) {
            std::cerr << "[ ERROR ] the weight file isn't yolo pose" << std::endl;
            exit(-1);
        }

        cv::Mat temp;
        inputTensors.clear();
        outputTensors.clear();
        if(!preprocessing(frame,temp)) return false;

        try{
            outputTensors = session.Run(Ort::RunOptions{nullptr},
                                        inputClasses.data(),
                                        inputTensors.data(),
                                        inputTensors.size(),
                                        outputClasses.data(),
                                        outputClasses.size());
        }catch(...){
            std::cout << "session Run failed." << std::endl;
            exit(-1);
        }

        auto* raw = outputTensors[0].GetTensorMutableData<float>();
        Person person;
        std::vector<Person> persons;
        std::vector<cv::Rect> rects;
        std::vector<float> confs;

        switch (version) {
            case PoseV5 :
            case PoseV8 : {
                int anchorsCount = (int)outputDims[2];
                const int x_offset = 0;
                const int y_offset = anchorsCount;
                const int width_offset = 2 * anchorsCount;
                const int height_offset = 3 * anchorsCount;
                const int confidence_offset = 4 * anchorsCount;
                const int pose_offset = 5 * anchorsCount;
                const int pose_x_offset = 0;
                const int pose_y_offset =  anchorsCount;
                const int pose_value_offset = 2 * anchorsCount;
                for(int i=0;i<anchorsCount;i++){
                    if(raw[confidence_offset + i] > confThreshold){
                        person.box.x = (int)(raw[x_offset + i] - raw[width_offset + i] / 2);
                        person.box.y = (int)(raw[y_offset + i] - raw[height_offset + i] / 2);
                        person.box.width = (int)raw[width_offset + i];
                        person.box.height = (int)raw[height_offset + i];
                        scaleCoords({frame.cols, frame.rows}, person.box);
                        auto kps = raw + pose_offset;
                        for(int k = 0;k < 17;k++){
                            switch (k) {
                                case 0 :POSE_PREPARE(person.nose);break;
                                case 1 :POSE_PREPARE(person.leftEye);break;
                                case 2 :POSE_PREPARE(person.rightEye);break;
                                case 3 :POSE_PREPARE(person.leftEar);break;
                                case 4 :POSE_PREPARE(person.rightEar);break;
                                case 5 :POSE_PREPARE(person.leftShoulder);break;
                                case 6 :POSE_PREPARE(person.rightShoulder);break;
                                case 7 :POSE_PREPARE(person.leftElbow);break;
                                case 8 :POSE_PREPARE(person.rightElbow);break;
                                case 9 :POSE_PREPARE(person.leftWrist);break;
                                case 10 :POSE_PREPARE(person.rightWrist);break;
                                case 11 :POSE_PREPARE(person.leftHip);break;
                                case 12 :POSE_PREPARE(person.rightHip);break;
                                case 13 :POSE_PREPARE(person.leftKnee);break;
                                case 14 :POSE_PREPARE(person.rightKnee);break;
                                case 15 :POSE_PREPARE(person.leftAnkle);break;
                                case 16 :POSE_PREPARE(person.rightAnkle);break;
                                default:break;
                            }
                        }
                        confs.emplace_back(raw[confidence_offset + i]);
                        rects.emplace_back(person.box);
                        persons.emplace_back(person);
                    }
                }
                std::vector<int> results;
                cv::dnn::NMSBoxes(rects,confs,confThreshold,iouThreshold,results);
                result.clear();
                for(const auto& num : results){
                    result.emplace_back(persons[num]);
                }
                return true;
            }
            default : {
                std::cerr << "[ ERROR ] the weight file isn't yolo pose" << std::endl;
                exit(-1);
            }
        }

    }

    Yolo::~Yolo() {
        for(auto& name : inputClasses) { delete[] name; name = nullptr; }
        for(auto& name : outputClasses) { delete[] name; name = nullptr; }
    }

    void Yolo::letterbox(const cv::Mat &frame, cv::Mat &outImage, const cv::Scalar &color, bool auto_, bool scaleFill,
                         bool scaleUp, int stride) const {
        cv::Size shape = frame.size();
        float r = std::min((float)size.height / (float)shape.height,
                           (float)size.width / (float)shape.width);
        if (!scaleUp)
            r = std::min(r, 1.0f);

        float ratio[2] {r, r};
        int newUnpad[2] {(int)std::round((float)shape.width * r),
                         (int)std::round((float)shape.height * r)};

        auto dw = (float)(size.width - newUnpad[0]);
        auto dh = (float)(size.height - newUnpad[1]);

        if (auto_)
        {
            dw = (float)((int)dw % stride);
            dh = (float)((int)dh % stride);
        }
        else if (scaleFill)
        {
            dw = 0.0f;
            dh = 0.0f;
            newUnpad[0] = size.width;
            newUnpad[1] = size.height;
            ratio[0] = (float)size.width / (float)shape.width;
            ratio[1] = (float)size.height / (float)shape.height;
        }

        dw /= 2.0f;
        dh /= 2.0f;

        if (shape.width != newUnpad[0] || shape.height != newUnpad[1])
        {
            cv::resize(frame, outImage, cv::Size(newUnpad[0], newUnpad[1]));
        }else{
            outImage = frame.clone();
        }

        int top = int(std::round(dh - 0.1f));
        int bottom = int(std::round(dh + 0.1f));
        int left = int(std::round(dw - 0.1f));
        int right = int(std::round(dw + 0.1f));
        cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    }

    void Yolo::scaleCoords(const cv::Size& originalSize, cv::Rect &coords) const {
        float gain = std::min((float)size.height / (float)originalSize.height,
                              (float)size.width / (float)originalSize.width);

        int pad[2] = {(int) (( (float)size.width - (float)originalSize.width * gain) / 2.0f),
                      (int) (( (float)size.height - (float)originalSize.height * gain) / 2.0f)};

        coords.x = (int) std::round(((float)(coords.x - pad[0]) / gain));
        coords.y = (int) std::round(((float)(coords.y - pad[1]) / gain));

        coords.width = (int) std::round(((float)coords.width / gain));
        coords.height = (int) std::round(((float)coords.height / gain));
    }

    void Yolo::scaleCoords(const cv::Size& originalSize, cv::Point &coords) const {
        float gain = std::min((float)size.height / (float)originalSize.height,
                              (float)size.width / (float)originalSize.width);
        int pad[2] = {(int) (( (float)size.width - (float)originalSize.width * gain) / 2.0f),
                      (int) (( (float)size.height - (float)originalSize.height * gain) / 2.0f)};
        coords.x = (int) std::round(((float)(coords.x - pad[0]) / gain));
        coords.y = (int) std::round(((float)(coords.y - pad[1]) / gain));
    }

    void Yolo::PaintDetections(
            cv::Mat                                     &frame,
            const std::vector<Detection>                &detections,
            const std::unordered_map<int,cv::Scalar>    &colorScheme
    ) {
        for(const auto& detection : detections) {
            auto it = colorScheme.find(detection.classId);
            cv::Scalar scalar;
            if(it == colorScheme.end()) scalar = {72,118,255};
            else scalar = it->second;
            cv::rectangle(frame, detection.box, scalar, 2);
            int x = detection.box.x;
            int y = detection.box.y;

            int conf = (int)std::round(detection.conf * 100);
            int classId = detection.classId;
            std::string label;
            if(outputNames.size() > classId){
                label = outputNames[classId] + " 0." + std::to_string(conf);
            }

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 2, &baseline);
            cv::rectangle(frame,
                          cv::Point(x, y - 25), cv::Point(x + textSize.width, y),
                          scalar, -1);

            cv::putText(frame, label,
                        cv::Point(x, y - 3), cv::FONT_ITALIC,
                        0.8, cv::Scalar(0, 0, 0), 2);
        }
    }

    void Yolo::PaintPersons(
            cv::Mat                                     &frame,
            const std::vector<Person>                   &persons,
            const std::unordered_map<int,cv::Scalar>    &colorScheme
    ) {
        for(const auto& person : persons) {
            POSE_PAINT(person.nose,frame,0);
            POSE_PAINT(person.leftEye,frame,1);
            POSE_PAINT(person.rightEye,frame,2);
            POSE_PAINT(person.leftEar,frame,3);
            POSE_PAINT(person.rightEar,frame,4);
            POSE_PAINT(person.leftShoulder,frame,5);
            POSE_PAINT(person.rightShoulder,frame,6);
            POSE_PAINT(person.leftElbow,frame,7);
            POSE_PAINT(person.rightElbow,frame,8);
            POSE_PAINT(person.leftWrist,frame,9);
            POSE_PAINT(person.rightWrist,frame,10);
            POSE_PAINT(person.leftHip,frame,11);
            POSE_PAINT(person.rightHip,frame,12);
            POSE_PAINT(person.leftKnee,frame,13);
            POSE_PAINT(person.rightKnee,frame,14);
            POSE_PAINT(person.leftAnkle,frame,15);
            POSE_PAINT(person.rightAnkle,frame,16);

            POSE_LINE(person.nose,person.leftEye,frame);
            POSE_LINE(person.nose,person.rightEye,frame);
            POSE_LINE(person.leftEye,person.leftEar,frame);
            POSE_LINE(person.rightEye,person.rightEar,frame);
            POSE_LINE(person.leftEar,person.leftShoulder,frame);
            POSE_LINE(person.rightEar,person.rightShoulder,frame);
            POSE_LINE(person.leftShoulder,person.leftElbow,frame);
            POSE_LINE(person.rightShoulder,person.rightElbow,frame);
            POSE_LINE(person.leftElbow,person.leftWrist,frame);
            POSE_LINE(person.rightElbow,person.rightWrist,frame);
            POSE_LINE(person.leftShoulder,person.leftHip,frame);
            POSE_LINE(person.rightShoulder,person.rightHip,frame);
            POSE_LINE(person.leftHip,person.leftKnee,frame);
            POSE_LINE(person.rightHip,person.rightKnee,frame);
            POSE_LINE(person.leftKnee,person.leftAnkle,frame);
            POSE_LINE(person.rightKnee,person.rightAnkle,frame);
        }
    }

    bool Yolo::Reload(
            const std::string &weightFilePath,
            Yolo::YoloVersion version,
            cv::Size size,
            bool cuda,
            Yolo::DeviceID deviceId,
            float confThreshold,
            float iouThreshold
    )
    {
        session.release();
        Ort::AllocatorWithDefaultOptions allocator;
        if(cuda && OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions,deviceId)) {
            std::cerr << "[ WARN ] using cuda failed!" << std::endl;
        }
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session{env,weightFilePath.c_str(),sessionOptions};

        inputClasses.resize(session.GetInputCount());
        for(int i=0;i<inputClasses.size();i++){
            inputClasses[i] = new char[strlen(session.GetInputNameAllocated(i,allocator).get()) + 1];
            strcpy(inputClasses[i],session.GetInputNameAllocated(i,allocator).get());
        }
        outputClasses.resize(session.GetOutputCount());
        for(int i=0;i<outputClasses.size();i++){
            outputClasses[i] = new char[strlen(session.GetOutputNameAllocated(i,allocator).get()) + 1];
            strcpy(outputClasses[i],session.GetOutputNameAllocated(i,allocator).get());
        }

        width = (int)session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[2];
        height = (int)session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[3];
        channel = (int)session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1];

        inputDims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        outputDims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        std::string names = session.GetModelMetadata().LookupCustomMetadataMapAllocated("names",allocator).get();
        auto pos = names.find('\'');
        while(pos != std::string::npos) {
            auto newPos = names.find('\'',pos + 1);
            outputNames.emplace_back(names.substr(pos + 1,newPos - pos -1));
            pos = names.find('\'',newPos + 1);
        }
        return false;
    }

} // hzd