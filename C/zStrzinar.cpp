//
// Created by ziga on 8.5.2017.
//

#include "zStrzinar.h"

void printMat(std::string text,  cv::Mat matrika){
    std::cout << text << matrika << std::endl;
}

void printContour(std::string text, std::vector<cv::Point> contour){
    std::cout << text;
    int i;
    for (i=0; i<contour.size(); i++){
        std::cout << contour[i] << ", ";
    }
    std::cout << std::endl;
}

bool pointsEqual(cv::Point a, cv::Point b){
    bool out =  (a.x==b.x) && (a.y==b.y);
    std::cout << a << ", " << b << std::endl;
    if (out) {
                std::cout << "here!" << std::endl;
        };
    return out;
}