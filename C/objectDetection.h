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

void getEdgeAndObjectNoScaling(const cv::Mat &areas, const cv::Size original_frame_size);

void keepLargestBlob(const cv::Mat &src, cv::Mat &dst);

void extractTheLargestCurve(const cv::Mat &dT, std::vector<cv::Point> &points);

std::vector <float> getOptimalLineImage_constrained(cv::Mat LineXY, float delta);

std::vector <cv::Mat> extractBlobs(cv::Mat bw);

void suppressDetections(std::vector<std::vector<int> >& boundingBoxes, std::vector<float>& areas);
#endif //PKP2017_OBJECTDETECTION_H
