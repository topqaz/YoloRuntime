//
// Created by top on 2026/1/5.
//

#ifndef YOLORUNTIME_UTILS_H
#define YOLORUNTIME_UTILS_H
#include <opencv2/opencv.hpp>


class utils {
};
void drawMask(cv::Mat& img, const cv::Mat& mask, cv::Rect box, const cv::Scalar& color, float alpha = 0.5f);

#endif //YOLORUNTIME_UTILS_H