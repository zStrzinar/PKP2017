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
#include "utility.h"

struct object{
    std::vector <float> bounding_box;
    float area;
};

void getEdgeAndObjectNoScaling(const cv::Mat &areas, const cv::Size originalFrameSize, std::vector <object> &objects, cv::Mat &xy_subset, std::vector<object> &suppressedObjects, cv::Mat &sea_region);

void keepLargestBlob(cv::Mat &in, cv::Mat &dst);

void extractTheLargestCurve(const cv::Mat &in, std::vector<cv::Point> &points);

std::vector <float> getOptimalLineImage_constrained(cv::Mat LineXY, float delta);

std::vector <cv::Mat> extractBlobs(cv::Mat bw);

void suppressDetections(const std::vector<object>& originalObjects, std::vector<object> &suppressedObjects);

void pruneobjs(const cv::Mat &bbs, std::vector<std::vector <int> > selected, std::vector<object>& suppressedObjects );

void mergeByProximity(cv::Mat& bbs_out, std::vector<object> bbs, std::vector<std::vector <int> >& selected_out);

void suppress_detections(cv::Mat bbs_in, cv::Mat Mu_in, std::vector<bool> selected, cv::Mat &bbs_out);

std::vector<int> CC2pixels(cv::Mat CC);
void ind2sub(cv::Size sz, std::vector<int> ind, std::vector<int> &x, std::vector<int> &y);
#endif //PKP2017_OBJECTDETECTION_H
