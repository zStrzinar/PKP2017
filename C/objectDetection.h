//
// Created by ziga on 26.4.2017.
//

#ifndef PKP2017_OBJECTDETECTION_H
#define PKP2017_OBJECTDETECTION_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>

void getEdgeAndObjectNoScaling(const cv::Mat &P_edge, const cv::Size Im_size);

void keepLargestBlob(const cv::Mat &src, cv::Mat &dst);

void extractTheLargestCurve(cv::Mat &dT, std::vector<cv::Point> &points);

cv::Mat getOptimalLineImage_constrained(cv::Mat LineXY, float delta);

#endif //PKP2017_OBJECTDETECTION_H
