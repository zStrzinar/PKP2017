//
// Created by ziga on 15.4.2017.
//

#ifndef PKP2017_UTILITY_H
#define PKP2017_UTILITY_H

#define REDUCE_TO_ROW 0
#define REDUCE_TO_COL 1

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

//enum Colorspace { HSV, RGB, YCRCB, LAB, YCRS, NONE };
enum BsxFunOp { DIVIDE, TIMES, MINUS, PLUS };

cv::Mat Bsxfun(cv::Mat inputMat, cv::Mat bsxParam, BsxFunOp op = DIVIDE);
cv::Mat columnOperations(cv::Mat inputMat, cv::Mat param, BsxFunOp op);
cv::Mat rowOperations(cv::Mat inputMat, cv::Mat param, BsxFunOp op);
double prod(cv::Mat mat);

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

#endif //PKP2017_UTILITY_H
