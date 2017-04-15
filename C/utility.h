//
// Created by ziga on 15.4.2017.
//

#ifndef PKP2017_UTILITY_H
#define PKP2017_UTILITY_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

enum Colorspace { HSV, RGB, YCRCB, LAB, YCRS, NONE };
enum BsxFunOp { DIVIDE, TIMES };

void printHelp();
void loadPriorModelFromDisk(Colorspace colorSpace, std::vector<cv::Mat> &mix_Mu, std::vector<cv::Mat> &mix_Cov, std::vector<cv::Mat> &mix_w, std::vector<cv::Mat> &static_prec);
cv::Mat Bsxfun(cv::Mat inputMat, cv::Mat bsxParam, BsxFunOp op);
void mexFunction(Colorspace colorSpace, cv::Mat sizeMask, bool use_uniform_component, double p_unknown,
                 std::string type_of_em, double min_lik_delta, int maxEMsteps, cv::Mat &mix_w, cv::Mat &PI_i,
                 cv::Mat &data, cv::Mat &mix_Mu, cv::Mat &mix_Cov, cv::Mat mix_priorOverMeans_static_Mu, cv::Mat static_prec,
                 bool use_prior_on_mixture, double epsilon, cv::Mat &Q_sum_large, cv::Mat &mix_PI_i);
double prod(cv::Mat mat);
Colorspace ResolveColorspace(std::string color);
double GetUnknownWeightForTheFeatureModel(Colorspace type_colorspace,cv::Mat sizeMask, bool use_uniform_component);
void  GetConvolutionKernel(std::string type_of_em, cv::Mat sizeMask, cv::Mat& H_0, cv::Mat& H_1);
cv::Mat normpdf(cv::Mat x, cv::Mat mu, cv::Mat prec, cv::Mat sigma, double epsilon);
cv::Mat mergePd(cv::Mat mu_d, cv::Mat c_d, cv::Mat mu_0, cv::Mat c0,cv::Mat ic0);

#endif //PKP2017_UTILITY_H
