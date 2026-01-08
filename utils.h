//
// Created by top on 2026/1/5.
//

#ifndef YOLORUNTIME_UTILS_H
#define YOLORUNTIME_UTILS_H
#include <opencv2/opencv.hpp>
#if defined(_WIN32)
#define YOLO_API extern "C" __declspec(dllexport)
#elif defined(__GNUC__)
#define YOLO_API extern "C" __attribute__((visibility("default")))
#else
#define YOLO_API extern "C"
#endif

class utils {
};
void drawMask(cv::Mat& img, const cv::Mat& mask, cv::Rect box, const cv::Scalar& color, float alpha = 0.5f);

#endif //YOLORUNTIME_UTILS_H