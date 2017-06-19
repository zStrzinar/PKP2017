//
// Created by ziga on 26.4.2017.
//

#ifndef PKP2017_SHOWRESULTS_H
#define PKP2017_SHOWRESULTS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "objectDetection.h"

//enum Colorspace { HSV, RGB, YCRCB, LAB, YCRS, NONE };
//enum BsxFunOp { DIVIDE, TIMES };

void displayEdgeAndObjects( const cv::Mat &srcImg,
                            cv::Mat &dstImg,
                            cv::Mat sel_xy,
                            std::vector<object> objects,
                            cv::Mat masked_sea,
                            std::vector<cv::Mat> &current_mix_Mu,
                            std::vector<cv::Mat> &current_mix_Cov,
                            std::vector<cv::Mat> &current_mix_W,
                            int display_type);
void displayEdgeAndObjects1( const cv::Mat &srcImg,
                            cv::Mat &dstImg,
                            cv::Mat sel_xy,
                            std::vector<object> objects,
                            cv::Mat masked_sea,
                            cv::Mat &current_mix_Mu,
                            std::vector<cv::Mat> &current_mix_Cov,
                            cv::Mat &current_mix_W);
void displayEdgeAndObjects2( const cv::Mat &srcImg,
                             cv::Mat &dstImg,
                             cv::Mat sel_xy,
                             std::vector<object> objects,
                             cv::Mat masked_sea,
                             cv::Mat &current_mix_Mu,
                             std::vector<cv::Mat> &current_mix_Cov,
                             cv::Mat &current_mix_W);

cv::RotatedRect getErrorEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat);

void drawEdge(cv::Mat &Img, std::vector<cv::Point> edge, cv::Scalar color, int width);

#endif //PKP2017_SHOWRESULTS_H
