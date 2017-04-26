//
// Created by ziga on 26.4.2017.
//

#include "objectDetection.h"

using namespace cv;
using namespace std;

void getEdgeAndObjectNoScaling(const cv::Mat &P_edge, const cv::Size Im_size){
    cv::Size size_edge = Im_size;
    cv::Size size_obj(P_edge.cols, P_edge.rows);
    std::vector<float> scl;
    scl.push_back((float)size_edge.height/(float)size_obj.height);
    scl.push_back((float)size_edge.width/(float)size_obj.width);
    float Tt_data[4] = {scl[0],0.0,0.0,scl[1]};
    cv::Mat Tt(2, 2, CV_32F,Tt_data);

    // Sea edge and its uncertainty score

    // Find max value in P_edge for each pixel (max value of all channels in one position)
    cv::Mat T_max(size_obj,P_edge.type());
    std::vector <cv::Mat>P_edge_ch;
    cv::split(P_edge,P_edge_ch);

    T_max = cv::max(P_edge_ch[0], P_edge_ch[1]);
    T_max = cv::max(P_edge_ch[2], T_max);
    T_max = cv::max(P_edge_ch[3], T_max);

    cv::Mat T;
    cv::compare(T_max,P_edge_ch[2], T, cv::CMP_EQ);
    std::cout << T << std::endl;

    // TODO: detektiraj in ohrani samo največje območje v T!

//    // Setup SimpleBlobDetector parameters.
//    cv::SimpleBlobDetector::Params params;
//
//    // Change thresholds
//    params.minThreshold = 10;
//    params.maxThreshold = 200;
//
//    // Filter by Area.
//    params.filterByArea = true;
//    params.minArea = 10;
//
//    // Filter by Circularity
//    params.filterByCircularity = false;
//    params.minCircularity = 0.1;
//
//    // Filter by Convexity
//    params.filterByConvexity = false;
//    params.minConvexity = 0.87;
//
//    // Filter by Inertia
//    params.filterByInertia = false;
//    params.minInertiaRatio = 0.01;
//
//    SimpleBlobDetector detector(params);
//
//    // Storage for blobs
//    std::vector<KeyPoint> keypoints;
//
//    // Detect blobs
//    detector.detect( T, keypoints);
//
//    // Draw detected blobs as red circles.
//    // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
//    // the size of the circle corresponds to the size of blob
//
//    Mat im_with_keypoints;
//    drawKeypoints( T, keypoints, im_with_keypoints, Scalar(255,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
//
//    // Show blobs
//    imshow("keypoints", im_with_keypoints );
    waitKey(0);

}
