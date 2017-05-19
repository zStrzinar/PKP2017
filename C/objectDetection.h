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

struct object{
    std::vector <int> bounding_box;
    float area;
};

void getEdgeAndObjectNoScaling(const cv::Mat &areas, const cv::Size original_frame_size, std::vector<object>& objects);

void keepLargestBlob(const cv::Mat &src, cv::Mat &dst);

void extractTheLargestCurve(const cv::Mat &dT, std::vector<cv::Point> &points);

std::vector <float> getOptimalLineImage_constrained(cv::Mat LineXY, float delta);

std::vector <cv::Mat> extractBlobs(cv::Mat bw);

void suppressDetections(const std::vector<object>& originalObjects, std::vector<object> &suppressedObjects);

void pruneobjs(const std::vector<object>& originalObjects, std::vector<int> selected, std::vector<object>& suppressedObjects);

void mergeByProximity(std::vector<object> objects, std::vector<int>& kept);
#endif //PKP2017_OBJECTDETECTION_H
