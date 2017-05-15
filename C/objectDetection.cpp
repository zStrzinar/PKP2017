//
// Created by ziga on 26.4.2017.
//

#include "objectDetection.h"
#include "utility.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "zStrzinar.h"

using namespace cv;
using namespace std;

void getEdgeAndObjectNoScaling(const cv::Mat &areas, const cv::Size original_frame_size){
    // areas is 4-channel CV_8UC4 image. Each channel represents one area + 4th channel is ____
    // original_frame_size is original frame size
    if (areas.type()!=CV_8UC4){
        std::cerr << "areas.type() is not unsigned 8bit!" << std::endl;
        std::cout << "areas.type() is " << areas.type() << std::endl;
    }
    cv::Size size_edge = original_frame_size; // 460x640
    cv::Size size_obj(areas.cols, areas.rows); // 50x50
    std::vector<float> scl; // scaling vector
    scl.push_back((float)size_edge.height/(float)size_obj.height); // razmerje višine med frame in 50x50
    scl.push_back((float)size_edge.width/(float)size_obj.width); // razmerje širine med frame in 50x50
    float Tt_data[4] = {scl[0],0.0,0.0,scl[1]}; // podatki za diagonalno matriko Tt
    cv::Mat Tt(2, 2, CV_32F,Tt_data); // Diagonalna matrika za skaliranje

    // ************** Get sea edge and its uncertainty score ***********************************************************

    // Find max value in areas for each pixel (max value of all channels in one position)
    cv::Mat areas_max(size_obj,areas.type());
    std::vector <cv::Mat>areas_ch;
    cv::split(areas,areas_ch);

    areas_max = cv::max(areas_ch[0], areas_ch[1]);
    areas_max = cv::max(areas_ch[2], areas_max);
    areas_max = cv::max(areas_ch[3], areas_max);
    // areas_max je matrika največjih vrednosti po vseh pikslih.

    cv::Mat T; // T is sea region
    cv::Mat sea_region(areas_max.rows,areas_max.cols,CV_8UC1,Scalar::all(0)); // sea_region is the larges continuous sea region

    cv::compare(areas_max, areas_ch[2], T, cv::CMP_EQ); // TODO: is areas_ch[2] sea region?

    // poskrbi za 8 connectivity: TODO: a to že deluje???
    cv::morphologyEx(T,T,MORPH_CLOSE, cv::getStructuringElement(MORPH_RECT, cv::Size(3,3)));

    // keep only the largest continuous region
    keepLargestBlob(T,sea_region);
    sea_region=sea_region.t();

    cv::Mat T_transposed, T_diff, Tt_diff, firstCol, dT;
    cv::Mat dT2_diff, firstRow, dT2;
    T_diff = T.rowRange(1,T.rows)-T.rowRange(0,T.rows-1); // nadomestek diff(T) v MATLAB; Od vrstic 2...end odštejemo vrstice 1...end-1. Torej gledamo kako so se spremenile vrednosti med dvema vrsticama
    T_transposed = T.t(); // nadomestek T' v MATLAB
    Tt_diff = T_transposed.rowRange(1,T_transposed.rows)-T_transposed.rowRange(0,T_transposed.rows-1); // nadomestek diff(T') v MATLAB
    Tt_diff = Tt_diff.t(); // nadomestek diff(T')' v MATLAB

    // popravek velikosti (rezultat diff()) in threshold
    firstCol = Mat::zeros(Tt_diff.rows,1,Tt_diff.type()); // diff() odstrani eno vrstico zato bomo na začetek dodali eno vrstico ničel
    hconcat(firstCol,Tt_diff,dT); // dT je [ničle; rezultat diff()]
    cv::compare(dT, Mat::zeros(dT.rows,dT.cols,dT.type()),dT,CMP_NE); // dT je imel do zdaj vrednosti [0,...,255], zdaj pa rata samo {0,255} - vse kar je večje od 0 rata 255 (threshold)

    // isto naredimo še enkrat
    firstRow = Mat::zeros(1,T_diff.cols,T_diff.type());
    vconcat(firstRow, T_diff, dT2);
    cv::compare(dT2, Mat::zeros(dT2.rows, dT2.cols, dT2.type()), dT2, CMP_NE);

    dT = dT | dT2; // dT zdaj govori ali je prišlo do spremembe ali po vrsticah ali po stolpcih.

    cv::Mat Data;
    bool everythingIsSea = cv::countNonZero(T) == (T.rows*T.cols);
    bool noChangesInSea = cv::countNonZero(dT) == 0;
    if (not everythingIsSea && not noChangesInSea){
        std::vector<Point> contour;
        extractTheLargestCurve(dT, contour);

        Data = cv::Mat(2,(int)contour.size(),CV_32F);
        // contour is in "50x50" coordinate system. We must transform to frame original coordinates! (using Tt scaling matrix)
        for (int i=0; i<contour.size(); i++){ // gremo cez vse tocke v contours
            cv::Mat V; // matrix representation of contour point
            cv::Mat(contour[i], false).convertTo(V,CV_32F); // conversion from contour point to matrix
            Data.col(i) = Tt*V-Tt*Mat::ones(2,1,CV_32F)/2; // scaling
        }
        // Data is now a matrix of points in frame coordinates
        // The points in data represent the largest = longest curve = sea area boundary.
    }
    else {
        // V matlabu na tem mestu ustvarijo prazen objekt objects
        // TODO: sem se vrni ko boš vedel kako imaš shranjene objekte
        std::cerr << "Not supported!" << std::endl;
    }

    //  Edge of sea
    float delta;
    delta = areas.rows*(float)0.3; // iz MATLAB

    // ************ ?????????????????????????? **************************
    std::vector <float> a;
    a = getOptimalLineImage_constrained(Data, delta);
    cv::Mat xy_subset = Data.clone();

    float x0,y0,x1,y1;
    x0 = 0; // x0
    y0 = -(a[0]*x0 + a[2])/a[1];
    x1 = original_frame_size.width;
    y1 = -(a[0]*x1 + a[3])/a[1];
    std::vector <float> pts;
    pts.push_back(x0);
    pts.push_back(y0);
    pts.push_back(x1);
    pts.push_back(y1);
    // ************ ?????????????????????????? **************************

    // I=~T'
    cv::Mat I;
    I = cv::Mat::ones(T.cols,T.rows,CV_8U)-T.t(); // v ones() sem zamenjal cols in rows, ker je potem T transponiran!

    std::vector <cv::Mat> CC;
    CC = extractBlobs(I); // v CC so blobi vsak svoja matrika

    // iz blobov dobimo bounding boxe v originalnem (večjem kot 50x50) frame
    int i;
    std::vector<std::vector <int> > boundingBoxes; std::vector<float> box_areas;
    for (i=0; i<CC.size(); i++){
        // find bounding box in 50x50 image
        int col, row;
        int col_min = -1, col_max=-1, row_min=-1, row_max=-1;
        cv::Mat colsSum, rowsSum;
        cv::reduce(CC[i], colsSum, REDUCE_TO_ROW, CV_REDUCE_SUM, CV_32F); // reduce lahko vrne samo 32S, 32F ali 64F
        cv::reduce(CC[i], rowsSum, REDUCE_TO_COL, CV_REDUCE_SUM, CV_32F);
        // TODO: spremeni tip matrik CC[] v CV_8U
        for (col = 0; col<colsSum.cols; col++){ // TODO: lahko to nadomestiš z minMaxLoc?
            if(colsSum.at<char>(col)!=0){
                if(col_min ==-1) {
                    col_min = col;
                }
                col_max = col;
            }
        }
        for (row = 0; row<rowsSum.rows; row++){ // TODO: lahko to nadomestiš z minMaxLoc?
            if(rowsSum.at<char>(row)!=0){
                if(row_min ==-1) {
                    row_min = row;
                }
                row_max = row;
            }
        }

        row_min-=1; col_min -=1;
        int height, width;
        height = col_max-col_min; // X je višina
        width = row_max-row_min; // Y je širina

        // rescale bounding box with values from Tt
        row_min*=Tt.at<float>(0,0);
        col_min*=Tt.at<float>(1,1);
        width*=Tt.at<float>(0,0);
        height*=Tt.at<float>(1,1);

        // Make a boundingBox object - a vector!
        std::vector <int> boundingBox;
        boundingBox.push_back(row_min);
        boundingBox.push_back(col_min);
        boundingBox.push_back(width);
        boundingBox.push_back(height);

        float area = width*height;

        // To je zdaj samo za preverjanje ali naj bounding box dodamo na seznam (v vektor)?
        // Poiščemo piksel ki ima po y najmanjšo koordinato. (Poiščemo x in y koordinati v 50x50)
        // Transformiamo v 860x640
        col_min/=Tt.at<float>(0,0);
        // TODO: ali je tukaj namen da najdemo koordinato zgoraj levo??? za bounding box al kaj...
        // Tole zdaj poišče zadnjo vrstico, ki ima v stolpcu col_min vrednost različno od 0.
        for (row=0; row<CC[i].rows; row++){
            if(CC[i].at<char>(col_min, row)){ // TODO: nisem prepričan glede zaporedja col,row ali pa row,col
                row_min = row;
                // TODO: bi tukaj moral biti break?
            }
        }
        col_min*=Tt.at<float>(0,0); // skaliranje
        row_min*=Tt.at<float>(1,1);

        cv::Mat temp;
        temp = abs(xy_subset.row(0)-col_min); // ???
        cv::Point loc;
        cv::minMaxLoc(temp, NULL, NULL, &loc, NULL); // find min. locations
        cv::Mat boundry;
        boundry = xy_subset.col(loc.x);

        if (boundry.at<float>(0,1)>row_min){ // TODO: tukaj spet preveri kaj pomenita 0,1 ??
            continue;
        }

        // objekt je prestal test, dodajmo ga v vektor
        boundingBoxes.push_back(boundingBox);
        box_areas.push_back(area);
    }
    suppressDetections(boundingBoxes, box_areas);
}

void keepLargestBlob(const cv::Mat &src, cv::Mat &dst){
    // src is binary image (CV_8U) - 0 is background, 255 are objects
    // dst is binary image (CV_8U) of only the largest blob in src
    if (src.type()!=CV_8U) {
        std::cerr << "src must be a binary image! src.type() must be CV_8U!";
    };
    if (dst.type()!=CV_8U) {
        std::cerr << "dst must be a binary image! dst.type() must be CV_8U!";
    };
    vector<vector<Point> > contours; // Vector for storing contours
    vector<Vec4i> hierarchy; // required for function findContours()

    findContours( src, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image
    // find the blob with the largest area
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

    // mark the biggest blob in output image
    Scalar color( 255,255,255); // white color
    drawContours( dst, contours,largest_contour_index, color, CV_FILLED, 8, hierarchy ); // Draw the largest contour using previously stored index.
}

void extractTheLargestCurve(const cv::Mat &src, std::vector<cv::Point> &points){
    // src is binary image matrix (CV_8U)
    // points is output vector of points representing the largest curve
    if (src.type()!=CV_8U){
        std::cerr << "src.type() must be CV_8U! source image must be binary!" << std::endl;
    }

    // find all curves
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    cv::findContours( src, contours, hierarchy, CV_RETR_LIST , CV_CHAIN_APPROX_SIMPLE );

    // TODO: kaj naj se zgodi ko NI črt?
    if (contours.size() == 1){
        std::cerr << "Not-handled behaviour! No curves found!" << std::endl;
    }

    // find the longest curve
    double longest_length = -1;
    int longest_contour_index = -1;
    for (int i=0; i<contours.size(); i++){
        double length = contours[i].size();
        if (length > longest_length){
            longest_length = length;
            longest_contour_index = i;
        }
    }

    points = contours[longest_contour_index];
}

std::vector <float> getOptimalLineImage_constrained(cv::Mat LineXY, float delta){
    // TODO: kaj sploh dela ta funkcija?
    cv::Mat a;
    a = cv::Mat(3,1,CV_32F);

    // Izracunamo kovariancno matriko in jo ustrezno skaliramo (razlika med matlab in openCV!)
    cv::Mat covarMat, meanMat, LineXY_t;
    cv::calcCovarMatrix(LineXY,covarMat, meanMat, CV_COVAR_NORMAL|CV_COVAR_COLS|CV_COVAR_SCALE, CV_32F);
    covarMat = covarMat*LineXY.cols/(LineXY.cols-1);

    // Izvedemo SVD
    cv::Mat S,U,V;
    cv::SVD::compute(covarMat, S,U,V);

    // Prva dva elementa a-ja naj bosta kar drugi stolpec U-ja
    U.col(1).copyTo(a.rowRange(0,2));
    // Tretji element a-ja je: -U(:,2)*meanMat
    cv::Mat temp(1,1,CV_32F);
    temp = -a.rowRange(0,2).t()*meanMat;
    temp.copyTo(a.row(2));

    double minVal = 1e-10; cv::Mat val; // vrednost pri kateri naj se for zanka ustavi
    cv::Mat a0 = a.clone(); // da bomo lahko spremljali spremembe
    cv::Mat rr, zerovls, w(1,LineXY.cols, CV_32F), w_sqrt;
    float sum_w;
    cv::Mat xyw, mnxy, sub, wd;
    cv::Mat C, diff;

    int iter,i; // iter je samo za for-zanko, i je za vse notranje for-zanke
    for (iter=0; iter<5; iter++){
        rr = abs(a.at<float>(0)*LineXY.row(0) + a.at<float>(1)*LineXY.row(1) + a.at<float>(2))/delta;
        cv::compare(rr,2,zerovls,CMP_GT);

        // na mesta kjer je rr<=2 v w vpišemo nek eksponenten člen
        // sproti seštevamo w
        // kjer rr>2 vpišemo 0
        sum_w = 0;
        for (i=0; i<w.cols; i++){
            if(zerovls.at<char>(i)==0) {
                w.at<float>(i) = exp(-(float) 0.5 * rr.at<float>(i) * rr.at<float>(i));
                sum_w += w.at<float>(i);
            }
            else
                w.at<float>(i) = 0;
        }
        // Normiramo w (vsota elementov je 1)
        for (i=0; i<w.cols; i++){
            w.at<float>(i)/=sum_w;
        }

        // obe vrstici v LineXY pomnožimo z w (vrstični vektor)
        xyw = rowOperations(LineXY,w,TIMES);

        // potem pa naredimo vsoto skozi stolpce (rezultat je en stolpec)
        cv::reduce(xyw,mnxy,REDUCE_TO_COL, CV_REDUCE_SUM);

        // obema vrsticama v LineXY odštejemo dobljen seštevek (zgoraj)
        // korenimo
        sub = columnOperations(LineXY,mnxy,MINUS);
        cv::sqrt(w,w_sqrt);
        // odštete vrednosti po vrsticah pomnožimo s koreni
        wd = rowOperations(sub,w_sqrt,TIMES);

        // Še enkrat SVD
        C = wd*wd.t();
        cv::SVD::compute(C, S,U,V);
        cv::Mat minusU;
        minusU = U.col(1)*(-1);
        minusU.copyTo(U.col(1));

        // a je spet določen tako kot prej
        U.col(1).copyTo(a.rowRange(0,2));
        temp = -a.rowRange(0,2).t()*mnxy;
        temp.copyTo(a.row(2));

        // kakšna je sprememba a-ja v tej iteraciji?
        diff = abs(a0-a);
        cv::reduce(diff,val,0,CV_REDUCE_AVG,CV_32F);
        // če je premajhna break
        if(val.at<float>(0)<minVal) break;
        // naslednjič primerjamo z novim a-jem
        a0 = a.clone();
    }

    std::vector <float> out;
    out.push_back(a.at<float>(0));
    out.push_back(a.at<float>(1));
    out.push_back(a.at<float>(2));
    return out;
}

std::vector <cv::Mat> extractBlobs(cv::Mat bw){
    std::vector <cv::Mat> out;

    vector<vector<Point> > contours; // Vector for storing contours
    vector<Vec4i> hierarchy;

    cv::findContours( bw, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image

    int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        cv::Mat currentBlob;
        currentBlob = Mat::zeros(bw.rows,bw.cols,CV_8U);
        Scalar color( 255, 255, 255 );
        drawContours(currentBlob,contours,idx,color, CV_FILLED, 8, hierarchy);
        out.push_back(currentBlob.clone());
    }
    return out;
}

void suppressDetections(std::vector<std::vector<int> >& boundingBoxes, std::vector<float> &areas){

}