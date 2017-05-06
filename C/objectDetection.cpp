//
// Created by ziga on 26.4.2017.
//

#include "objectDetection.h"
#include "utility.h"

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

    cv::Mat T,T_largest(T_max.rows,T_max.cols,CV_8UC1,Scalar::all(0));
    cv::compare(T_max,P_edge_ch[2], T, cv::CMP_EQ);

    imshow("prej", T);
    // poskrbi za 8 connectivity:
    cv::morphologyEx(T,T,MORPH_CLOSE, cv::getStructuringElement(MORPH_RECT, cv::Size(3,3)));
    imshow("potem", T);

    // ohranimo samo največje območje
    keepLargestBlob(T,T_largest);
    T_largest=T_largest.t();
    imshow( "largest Contour", T_largest );

    cv::Mat T_transposed, T_diff,Tt_diff, firstCol, dT;
    cv::Mat dT2_diff, firstRow,dT2;
    T_diff = T.rowRange(1,T.rows)-T.rowRange(0,T.rows-1); // nadomestek diff(T)
    T_transposed = T.t(); // nadomestek T'
    Tt_diff = T_transposed.rowRange(1,T_transposed.rows)-T_transposed.rowRange(0,T_transposed.rows-1); // nadomestek diff(T')
    Tt_diff = Tt_diff.t(); // nadomestek diff(T')'

    firstCol = Mat::zeros(Tt_diff.rows,1,Tt_diff.type());
    hconcat(firstCol,Tt_diff,dT);
    cv::compare(dT, Mat::zeros(dT.rows,dT.cols,dT.type()),dT,CMP_NE);

    firstRow = Mat::zeros(1,T_diff.cols,T_diff.type());
    vconcat(firstRow, T_diff, dT2);
    cv::compare(dT2, Mat::zeros(dT2.rows, dT2.cols, dT2.type()), dT2, CMP_NE);

    dT = dT | dT2;
    cv::Mat Data;
    if ((cv::countNonZero(T) != (T.rows*T.cols)) && (cv::countNonZero(dT)!=0)){
        std::vector<Point> contour;
        extractTheLargestCurve(dT, contour);

        Data = cv::Mat(2,(int)contour.size(),CV_32F);
        for (int i=0; i<Data.cols; i++){
            cv::Mat V;
            cv::Mat(contour[i], false).convertTo(V,CV_32F);
            Data.col(i) = Tt*V-Tt*Mat::ones(2,1,CV_32F)/2;
        }
    }
    else {
        // V matlabu na tem mestu ustvarijo prazen objekt objects
        // TODO: sem se vrni ko boš vedel kako imaš shranjene objekte
    }

    std::cout << "dT: " << dT << std::endl;
    std::cout << "Data: " << Data << std::endl;

    //  Edge of sea
    float delta;
    delta = P_edge.rows*(float)0.3;

    getOptimalLineImage_constrained(Data, delta);

    imshow("dT", dT); waitKey();

}

void keepLargestBlob(const cv::Mat &src, cv::Mat &dst){
    vector<vector<Point> > contours; // Vector for storing contour
    vector<Vec4i> hierarchy;

    findContours( src, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image
    double largest_area=0;
    int largest_contour_index=0;
    for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
        double a=contourArea( contours[i],false);  //  Find the area of contour
        if(a>largest_area){
            largest_area=a;
            largest_contour_index=i;                //Store the index of largest contour
            }
    }

    Scalar color( 255,255,255);
    drawContours( dst, contours,largest_contour_index, color, CV_FILLED, 8, hierarchy ); // Draw the largest contour using previously stored index.
}

void extractTheLargestCurve(cv::Mat &dT, std::vector<cv::Point> &points){
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours( dT, contours, hierarchy, CV_RETR_LIST , CV_CHAIN_APPROX_SIMPLE );
    std::cout << "Number of contours found: " << contours.size() << std::endl;
    std::cout << "contours.size() = " << contours.size() << std::endl << "contours.max_size() = " << contours.max_size() << std::endl;
    double longest_length = -1;
    int longest_contour_index = -1;
    for (int i=0; i<contours.size(); i++){
        double length = contours[i].size();
        std::cout << "Contour[" << i << "] is size() " << length << std::endl;
        if (length > longest_length){
            longest_length = length;
            longest_contour_index = i;
        }
    }

    points = contours[longest_contour_index];

    Scalar color( 255,255,255); // TODO: debug
    cv::Mat img(50,50,CV_64F,0.0);
    drawContours( img, contours,longest_contour_index, color, 1, 8, hierarchy ); // Draw the largest contour using previously stored index.
    imshow("testno okno", img.t());
}

void getOptimalLineImage_constrained(cv::Mat LineXY, float delta){
    cv::Mat a;
    a = cv::Mat(3,1,CV_32F);

    cv::Mat covarMat, meanMat, LineXY_t;
    cv::calcCovarMatrix(LineXY,covarMat, meanMat, CV_COVAR_NORMAL|CV_COVAR_COLS|COVAR_SCALE, CV_32F);
    covarMat = covarMat*LineXY.cols/(LineXY.cols-1);
    cv::Mat S,U,V;
    cv::SVD::compute(covarMat, S,U,V);
    U.col(1).copyTo(a.rowRange(0,2));
    cv::Mat temp(1,1,CV_32F);
    temp = -a.rowRange(0,2).t()*meanMat;
    temp.copyTo(a.row(2));

    double minVal = 1e-10; cv::Mat val;
    cv::Mat a0 = a;
    cv::Mat rr, zerovls;
    cv::Mat w(1,LineXY.cols, CV_32F), w_sqrt;
    float sum_w = 0;
    cv::Mat xyw(LineXY.rows, LineXY.cols,CV_32F), mnxy, sub, wd;
    cv::Mat C, diff;

    int iter,i;
    for (iter=0; iter<5; iter++){
        rr = abs(a.at<float>(0)*LineXY.row(0) + a.at<float>(1)*LineXY.row(1) + a.at<float>(2))/delta;
        zerovls = rr>2;
        for (i=0; i<w.cols; i++){
            if(zerovls.at<float>(i)==0) {
                w.at<float>(i) = exp(-(float) 0.5 * rr.at<float>(i) * rr.at<float>(i));
                sum_w += w.at<float>(i);
            }
            else
                w.at<float>(i) = 0;
        }
        for (i=0; i<w.cols; i++){
            w.at<float>(i)/=sum_w;
        }

        xyw = Bsxfun(LineXY,w,TIMES);
        cv::reduce(xyw,mnxy,1, CV_REDUCE_SUM);

        sub = Bsxfun(LineXY,mnxy,MINUS);
        cv::sqrt(w,w_sqrt);
        wd = Bsxfun(sub,w_sqrt,TIMES);

        C = wd*wd.t();
        cv::SVD::compute(C, S,U,V);
        U.col(1).copyTo(a.rowRange(0,2));
        temp = -a.rowRange(0,2).t()*mnxy;
        temp.copyTo(a.row(2));

        diff = abs(a0-a);
        cv::reduce(diff,val,0,CV_REDUCE_AVG,CV_32F);
        if(val.at<float>(0)<minVal) break;
        a0 = a.clone();
    }
    waitKey();
}