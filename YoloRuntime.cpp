#if defined(_WIN32)
#define YOLO_API extern "C" __declspec(dllexport)
#elif defined(__GNUC__)
#define YOLO_API extern "C" __attribute__((visibility("default")))
#else
#define YOLO_API extern "C"
#endif


#include <iostream>
#include "inference.h"
#include <utils.h>

typedef void* YOLO_HANDLE;


YOLO_API
YOLO_HANDLE CreateModel(const char *modelPath, int size, int task, bool useGPU, bool nms, bool half)
{
    YOLO_V8* detector = new YOLO_V8();
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.1;
    params.iouThreshold = 0.5;
    params.modelPath = modelPath;
    params.imgSize = {size, size};
    if (task == 0) params.task = DETECT;
    else if (task == 1) params.task = SEGMENT;
    else if (task == 2) params.task = POSE;
    else if (task == 3) params.task = OBB;
    params.nms = nms;
    params.cudaEnable = useGPU;
    params.half = half;
    detector->CreateSession(params);
    // try {
    //     if (!detector->CreateSession(params)) {
    //         throw std::runtime_error("CreateSession returned false");
    //     }
    // }
    // catch (const std::exception& e) {
    //     std::cerr << e.what() << std::endl;
    //     delete detector;
    //     return nullptr;
    // }
    return reinterpret_cast<YOLO_HANDLE>(detector);
}
YOLO_API
bool DetectImage(
        YOLO_HANDLE handle,
        const unsigned char *imgData,
        int width,
        int height,
        int stride,
        DetectionResult *results,
        int maxCount,
        int *outCount,
        bool saveResult
) {
    if (!handle || !imgData || !results || !outCount)
        return false;

    YOLO_V8* detector = reinterpret_cast<YOLO_V8*>(handle);

    try {
        if (stride < width * 3) return false;

        cv::Mat img(height, width, CV_8UC3, (void *)imgData, stride);
        cv::Mat imgCopy = img.clone();

        std::vector<DL_RESULT> res;
        detector->RunSession(imgCopy, res);

        int count = std::min((int)res.size(), maxCount);
        for (int i = 0; i < count; ++i) {

            if (!res[i].mask.empty() && res[i].mask.size() == imgCopy.size()) {
                static cv::RNG rng(12345);
                cv::Scalar color(rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));
                drawMask(imgCopy, res[i].mask, res[i].box, color, 0.8f);

            }

            cv::rectangle(imgCopy, res[i].box, cv::Scalar(0,255,0), 2);

            results[i].x = res[i].box.x;
            results[i].y = res[i].box.y;
            results[i].w = res[i].box.width;
            results[i].h = res[i].box.height;
            results[i].score = res[i].confidence;
            results[i].classId = res[i].classId;
        }

        if (saveResult) {
            cv::imwrite("res.jpg", imgCopy);
        }

        *outCount = count;
        return true;
    }
    catch (const std::exception&) {
        return false;
    }
}
YOLO_API
void DestroyModel(YOLO_HANDLE handle)
{
    if (!handle) return;
    YOLO_V8* detector = reinterpret_cast<YOLO_V8*>(handle);
    delete detector;
}
