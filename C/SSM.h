//
// Created by ziga on 5.7.2017.
//

#ifndef PKP2017_SSM_H
#define PKP2017_SSM_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


enum Colorspace { HSV, RGB, YCRCB, LAB, YCRS, NONE };
//enum BsxFunOp { DIVIDE, TIMES, MINUS, PLUS };


void run_SSM(Colorspace colorSpace,
             cv::Size sizeMask,
             bool use_uniform_component,
             std::string type_of_em,
             int maxEMsteps,
             std::vector<cv::Mat> &mix_w,
             cv::Mat &PI_i,
             cv::Mat &data,
             std::vector<cv::Mat> &mix_Mu,
             std::vector<cv::Mat> &mix_Cov,
             std::vector<cv::Mat> prior_mix_Mu,
             std::vector<cv::Mat> prior_mix_Prec,
             bool use_prior_on_mixture,
             long double epsilon,
             cv::Mat &Q_sum_large,
             cv::Mat &mix_PI_i);
Colorspace ResolveColorspace(std::string color);
double GetUnknownWeightForTheFeatureModel(Colorspace type_colorspace,cv::Size sizeMask, bool use_uniform_component);
void  GetConvolutionKernel(std::string type_of_em, cv::Size sizeMask, cv::Mat& H_0, cv::Mat& H_1);
cv::Mat normpdf(cv::Mat x, cv::Mat mu, cv::Mat prec, cv::Mat sigma, double epsilon);
cv::Mat mergePd(cv::Mat mu_d, cv::Mat c_d, cv::Mat mu_0, cv::Mat c0,cv::Mat ic0);
void loadPriorModelFromDisk(Colorspace colorSpace, std::vector<cv::Mat> &mix_Mu, std::vector<cv::Mat> &mix_Cov, std::vector<cv::Mat> &mix_w, std::vector<cv::Mat> &static_prec);
void getSpacialData(cv::Size em_image_size, cv::Mat& spatial_data);
void momentMatchPdf(cv::Mat previous_Mu, cv::Mat current_Mu, cv::Mat previous_Cov, cv::Mat current_Cov, std::vector<float> current_w, cv::Mat& new_Mu, cv::Mat& new_Cov, cv::Mat& new_w);


#endif //PKP2017_SSM_H
