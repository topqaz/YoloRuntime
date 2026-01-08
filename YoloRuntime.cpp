#include <iostream>
#include "inference.h"
#include <utils.h>

typedef void* YOLO_HANDLE;

YOLO_API
YOLO_HANDLE CreateModel(const char *modelPath, int size, int task, bool useGPU, bool nms, bool halfEnable)
{
    YOLO* detector = new YOLO();
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
    params.halfEnable = halfEnable;
    try {
        if (!detector->CreateSession(params)) {
            throw std::runtime_error("CreateSession returned false");
        }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        delete detector;
        return nullptr;
    }
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
        bool saveResult,
        imagedata *outImgData
) {
    if (!handle || !imgData || !results || !outCount)
        return false;

    YOLO* detector = reinterpret_cast<YOLO*>(handle);

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


        size_t imgSize = imgCopy.total() * imgCopy.elemSize();

        outImgData->data = new unsigned char[imgSize];
        memcpy(outImgData->data, imgCopy.data, imgSize);

        outImgData->width = imgCopy.cols;
        outImgData->height = imgCopy.rows;
        outImgData->channels = imgCopy.channels();

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
    YOLO* detector = reinterpret_cast<YOLO*>(handle);
    delete detector;
}
YOLO_API
void FreeData(unsigned char* p)
{
    if (p) delete[] p;
}

YOLO_API
bool CheckGpuAvailable()
{
#ifdef USE_CUDA
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gpu_check");

        Ort::SessionOptions so;
        OrtCUDAProviderOptions cuda_options;
        so.AppendExecutionProvider_CUDA(cuda_options);
        return true;
    }
    catch (...) {
        return false;
    }
#else
    return false;
#endif
}