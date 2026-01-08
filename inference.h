// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#define    RET_OK nullptr

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

struct DetectionResult {
    int x;
    int y;
    int w;
    int h;
    float score;
    int classId;
};

struct imagedata {
    unsigned char* data;
    int width;
    int height;
    int channels;
};



enum TASK
{
    DETECT = 0,
    SEGMENT = 1,
    POSE = 2,
    OBB = 3
};
typedef struct _DL_INIT_PARAM
{
    std::string modelPath;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.5;
    int	keyPointsNum = 2;//Note:kpt number for pose
    bool cudaEnable = false;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
    bool halfEnable = false;
    bool nms = false;
    TASK task = DETECT;
} DL_INIT_PARAM;


typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
    cv::Mat mask;
} DL_RESULT;


class YOLO
{
public:
    YOLO();

    ~YOLO();

public:
    bool CreateSession(DL_INIT_PARAM& iParams);

    bool RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);

    bool WarmUpSession();

    template<typename N>
    bool TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        std::vector<DL_RESULT>& oResult);

    bool PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);


private:
    Ort::Env env;
    Ort::Session* session;
    bool cudaEnable;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;

    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    float resizeScales;//letterbox scale
    bool halfEnable;
    bool nms;
    TASK task;
};
