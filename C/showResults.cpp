//
// Created by ziga on 26.4.2017.
//

#include "showResults.h"
#include "zStrzinar.h"

using namespace cv;
using namespace std;

void displayEdgeAndObjects( const cv::Mat &srcImg,
                            cv::Mat &dstImg,
                            cv::Mat sel_xy,
                            std::vector<object> objects,
                            cv::Mat masked_sea,
                            std::vector<cv::Mat> &current_mix_Mu,
                            std::vector<cv::Mat> &current_mix_Cov,
                            std::vector<cv::Mat> &current_mix_W,
                            int display_type) {
    cv::Mat T(2, 2, CV_64F, cv::Scalar(0));
    T.at<double>(0, 0) = (double) srcImg.cols / (double) masked_sea.cols;
    T.at<double>(1, 1) = (double) srcImg.rows / (double) masked_sea.rows;

    cv::Mat mixture_spatial_Mu, mixture_spatial_W;
    std::vector<cv::Mat> mixture_spatial_Cov;

    cv::Mat new_column;
    new_column = T * current_mix_Mu[0].rowRange(0, 2);
    mixture_spatial_Mu = new_column.clone();
    mixture_spatial_W = current_mix_W[0].clone();
    new_column = T * current_mix_Cov[0].rowRange(0, 2).colRange(0, 2) * T;
    mixture_spatial_Cov.push_back(new_column);

    int i;
    for (i = 1; i < current_mix_Mu.size(); i++) {
        new_column = T * current_mix_Mu[i].rowRange(0, 2);
        hconcat(mixture_spatial_Mu, new_column, mixture_spatial_Mu);

        hconcat(mixture_spatial_W, current_mix_W[i], mixture_spatial_W);

        new_column = T * current_mix_Cov[i].rowRange(0, 2).colRange(0, 2) * T;
        mixture_spatial_Cov.push_back(new_column);
    }

    switch (display_type) {
    case 1: {
        displayEdgeAndObjects1(srcImg, dstImg, sel_xy, objects, masked_sea, mixture_spatial_Mu, mixture_spatial_Cov,
                               mixture_spatial_W);
        break;
    }
    case 2: {
        std::cerr << "Display type 2 is not yet supported!" << std::endl;
    }
    default : {
        std::cerr << "Unsupported display type!" << std::endl;
    }
}

}

void displayEdgeAndObjects1( const cv::Mat &srcImg,
                             cv::Mat &dstImg,
                             cv::Mat sel_xy,
                             std::vector<object> objects,
                             cv::Mat masked_sea,
                             cv::Mat &current_mix_Mu,
                             std::vector<cv::Mat> &current_mix_Cov,
                             cv::Mat &current_mix_W){
    cv::Mat Image_plus, Regions;
    Image_plus = srcImg.clone();

    int i;
    for(i=0;i<3;i++){
        cv::Point2f mean((float)current_mix_Mu.col(i).at<double>(0),(float)current_mix_Mu.col(i).at<double>(1));
        cv::RotatedRect ellipse = getErrorEllipse(2.4477,mean,current_mix_Cov[i]);

        cv::ellipse(Image_plus, ellipse,cv::Scalar::all(255),2);
    }

    namedWindow("Results",CV_WINDOW_AUTOSIZE);
    cv::imshow("Results",Image_plus);
    waitKey(1);
}

cv::RotatedRect getErrorEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat){

    //Get the eigenvalues and eigenvectors
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(covmat, true, eigenvalues, eigenvectors);

    //Calculate the angle between the largest eigenvector and the x-axis
    double angle = atan2(eigenvectors.at<double>(0,1), eigenvectors.at<double>(0,0));

    //Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
    if(angle < 0)
        angle += 6.28318530718;

    //Conver to degrees instead of radians
    angle = 180*angle/3.14159265359;

    //Calculate the size of the minor and major axes
    double halfmajoraxissize=chisquare_val*sqrt(eigenvalues.at<double>(0));
    double halfminoraxissize=chisquare_val*sqrt(eigenvalues.at<double>(1));

    //Return the oriented ellipse
    //The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
    return cv::RotatedRect(mean, cv::Size2f(halfmajoraxissize, halfminoraxissize), -angle);

}
