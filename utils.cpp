//
// Created by top on 2026/1/5.
//

#include "utils.h"

void drawMask(cv::Mat& img, const cv::Mat& mask, cv::Rect box, const cv::Scalar& color, float alpha)
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
