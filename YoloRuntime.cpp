#if defined(_WIN32)
#define YOLO_API extern "C" __declspec(dllexport)
#elif defined(__GNUC__)
#define YOLO_API extern "C" __attribute__((visibility("default")))
#else
#define YOLO_API extern "C"
#endif

#include <iostream>
#include "inference.h"



void drawMask(cv::Mat& img, const cv::Mat& mask, cv::Rect box, const cv::Scalar& color, float alpha = 0.5f)
{
    cv::Rect validBox = box & cv::Rect(0, 0, img.cols, img.rows);
    if (validBox.area() == 0) return;
    cv::Mat roi = img(validBox);
    cv::Mat maskRoi = mask(validBox);
    cv::Mat binMask;
    if (maskRoi.type() != CV_8U) {
        cv::threshold(maskRoi, binMask, 0.5, 255, cv::THRESH_BINARY);
        binMask.convertTo(binMask, CV_8U);
    } else {
        binMask = maskRoi;
    }
    cv::Mat colorLayer(validBox.size(), CV_8UC3, color);

    // 5. 加权混合 (核心步骤)
    // blended = roi * (1-alpha) + colorLayer * alpha
    cv::Mat blended;
    cv::addWeighted(roi, 1.0 - alpha, colorLayer, alpha, 0.0, blended);
    blended.copyTo(roi, binMask);
}


static YOLO_V8 *g_yolodetect = nullptr;
YOLO_API
bool InitModel(const char *modelPath, int size,int task, bool useGPU,bool nms,bool half) {
    g_yolodetect = new YOLO_V8;
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.1;
    params.iouThreshold = 0.5;
    params.modelPath = modelPath;
    params.imgSize = {size, size};
    if(task==0)params.task = DETECT;
    else if(task==1)params.task = SEGMENT;
    else if(task==2)params.task = POSE;
    else if(task==3)params.task = OBB;
    params.nms = nms;
    if (useGPU) {
        params.cudaEnable = true;
        params.half = half;
    } else {
        params.half = half;
        params.cudaEnable = false;
    }
    std::cout << "modelPath:" << modelPath << std::endl;
    std::cout << "size:" << size << std::endl;
    std::cout << "useGPU:" << useGPU << std::endl;
    std::cout << "task:" << task << std::endl;
    std::cout << "nms:" << nms << std::endl;
    std::cout << "half:" << half << std::endl;

    g_yolodetect->CreateSession(params);
    return true;
}
YOLO_API
bool DetectImage(
        const unsigned char *imgData,
        int width,
        int height,
        int stride,
        DetectionResult *results,
        int maxCount,
        int *outCount,
        bool showResult
) {
    if (!g_yolodetect || !imgData || !results || !outCount)
        return false;
    try {
        cv::Mat img(height, width, CV_8UC3, (void *) imgData, stride);
        cv::Mat imgCopy = img.clone();
        std::cout << "img size:" << imgCopy.size() << std::endl;
        std::vector<DL_RESULT> res;
        g_yolodetect->RunSession(imgCopy, res);
        std::cout << "Detected " << res.size() << " objects." << "success" << std::endl;

        int count = std::min((int) res.size(), maxCount);
        for (int i = 0; i < count; ++i) {
            if (!res[i].mask.empty()){
            cv::RNG rng(cv::getTickCount());
            cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            drawMask(imgCopy, res[i].mask, res[i].box, color, 0.8);
        }
            cv::rectangle(imgCopy, res[i].box, cv::Scalar(0, 255, 0), 2);
            float confidence = floor(100 * res[i].confidence) / 100;
            std::string label = std::to_string(res[i].classId) + " " +
                                std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);
            cv::putText(imgCopy, label, cv::Point(res[i].box.x, res[i].box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);


            results[i].x = res[i].box.x;
            results[i].y = res[i].box.y;
            results[i].w = res[i].box.width;
            results[i].h = res[i].box.height;
            results[i].score = res[i].confidence;
            results[i].classId = res[i].classId;
        }
        if (showResult){
        cv::resize(imgCopy,imgCopy,cv::Size(800,800*height/width));
        cv::imshow("Detection Result", imgCopy);
        cv::waitKey(0);
    }
        *outCount = count;
        return true;
    }
    catch (...) {
        return false;
    }
}
YOLO_API
void ReleaseModel() {
    if (g_yolodetect) {
        delete g_yolodetect;
        g_yolodetect = nullptr;
    }
}
