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

void removeRow(const cv::Mat &in, cv::Mat &out, int index);
void removeCol(const cv::Mat &in, cv::Mat &out, int index);
void removeRows(const cv::Mat &in, cv::Mat &out, std::vector<bool> deleteVector);
void removeRows(const cv::Mat &in, cv::Mat &out, std::vector<int> deleteVector);
void removeCols(const cv::Mat &in, cv::Mat &out, std::vector<bool> deleteVector);
void removeCols(const cv::Mat &in, cv::Mat &out, std::vector<int> deleteVector);
template <typename T> // TODO: to nekako ne dela
void removeVectorElements(const std::vector<T> &in, std::vector<T> &out, std::vector<bool> deleteVector);
void removeVectorElementsInt(const std::vector<int> &in, std::vector<int> &out, std::vector<bool> deleteVector);
void removeCirclebackY(std::vector<cv::Point> &contour);
void firstLastIdx(cv::Mat input, int& firstIdx, int& lastIdx);
void myBGR2YCrCb(cv::Mat BGR, cv::Mat& YCrCb);
void myMinMaxValIdx(std::vector<float> a,float &minVal, float &maxVal, int &minIdx,int &maxIdx);
#endif //PKP2017_ZSTRZINAR_H
