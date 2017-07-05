//
// Created by ziga on 8.5.2017.
//

#ifndef PKP2017_ZSTRZINAR_H
#define PKP2017_ZSTRZINAR_H
#include <iostream>
#include <opencv2/core/core.hpp>
void printMat(std::string,  cv::Mat matrika);
void printContour(std::string text, std::vector<cv::Point> contour);
bool pointsEqual(cv::Point a, cv::Point b);
#endif //PKP2017_ZSTRZINAR_H
