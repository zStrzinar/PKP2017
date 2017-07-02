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

void getEdgeAndObjectNoScaling(const cv::Mat &areas, const cv::Size originalFrameSize, std::vector <object> &objects, cv::Mat &xy_subset, std::vector<object> &suppressedObjects, cv::Mat &sea_region){ // TODO: v Matlabu se poleg objektov vrnejo še xy koordinate nečesa in masked_sea
    // areas is 4-channel CV_64FC4 image. Each channel represents one area + 4th channel is ____
    // original_frame_size is original frame size
    if (areas.type()!=CV_64FC4){
        std::cerr << "areas.type() is not 64bit float!" << std::endl;
        std::cout << "areas.type() is " << areas.type() << std::endl;
    }

    cv::Size size_obj(areas.cols, areas.rows); // 50x50
    std::vector<float> scl; // scaling vector
    scl.push_back((float)originalFrameSize.width/(float)size_obj.width); // razmerje višine med frame in 50x50
    scl.push_back((float)originalFrameSize.height/(float)size_obj.height); // razmerje širine med frame in 50x50
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

    // V T je 1(255) tam kjer je bila največja vrednost vseh pikslov ravno za morje (tretji kanal od štirih)
    cv::Mat T; // T is sea region
    sea_region = cv::Mat(areas_max.rows,areas_max.cols,CV_8UC1,Scalar::all(0)); // sea_region is the largest continuous sea region


    cv::compare(areas_max, areas_ch[2], T, cv::CMP_EQ);
    // TODO: odstrani 8-connectivity ozadja v T

    // keep only the largest continuous region
    cv::Mat temp; temp = T.clone();

    keepLargestBlob(T,sea_region); // To zdaj deluje


    T = sea_region.clone();
    // v MATLABU je tukaj še sea_region=~bwmorph(~sea_region,'clean'), kar odstrani osamele piksle. Tega itak ni ker smo ohrnili samo največjo regijo!
    sea_region=sea_region.t(); // To je v redu

    cv::Mat T_transposed, T_diff, Tt_diff, firstCol, dT;
    cv::Mat dT2_diff, firstRow, dT2;
    T_diff = (T.rowRange(1,T.rows)-T.rowRange(0,T.rows-1))|(T.rowRange(0,T.rows-1)-T.rowRange(1,T.rows)); // nadomestek diff(T) v MATLAB; Od vrstic 2...end odštejemo vrstice 1...end-1. Torej gledamo kako so se spremenile vrednosti med dvema vrsticama. TODO: poštimaj da ne bo več opozorila za neujemanje tipov!
    // Ta moj način delanja diff() je malo drugačen od matlaba. V matlabu to pomeni res koliko je razlike med dvema sosednjima elementoma. Jaz pa samo gleda ALI je prišlo do spremembe? Jaz imam torej matriko booleanov! Ampak za delo naprej nam to ustreza!

    T_transposed = T.t(); // nadomestek T' v MATLAB
    Tt_diff = (T_transposed.rowRange(1,T_transposed.rows)-T_transposed.rowRange(0,T_transposed.rows-1))|(T_transposed.rowRange(0,T_transposed.rows-1)-T_transposed.rowRange(1,T_transposed.rows)); // nadomestek diff(T') v MATLAB - glej razlago zgoraj! TODO: poštimaj da ne bo več opozorila za neujemanje tipov!
    Tt_diff = Tt_diff.t(); // nadomestek diff(T')' v MATLAB

    // popravek velikosti (rezultat diff()) in threshold
    firstCol = Mat::zeros(Tt_diff.rows,1,Tt_diff.type()); // diff() odstrani eno vrstico zato bomo na začetek dodali eno vrstico ničel
    hconcat(firstCol,Tt_diff,dT); // dT je [ničle; rezultat diff()]
    cv::compare(dT, Mat::zeros(dT.rows,dT.cols,dT.type()),dT,CMP_NE); // dT je imel do zdaj vrednosti [0,...,255], zdaj pa rata samo {0,255} - vse kar je večje od 0 rata 255 (threshold)

    // isto naredimo še enkrat
    firstRow = Mat::zeros(1,T_diff.cols,T_diff.type());
    vconcat(firstRow, T_diff, dT2);
    cv::compare(dT2, Mat::zeros(dT2.rows, dT2.cols, dT2.type()), dT2, CMP_NE);
    cv::bitwise_or(dT,dT2,dT); // dT zdaj govori ali je prišlo do spremembe ali po vrsticah ali po stolpcih.

    cv::Mat Data;
    bool everythingIsSea = cv::countNonZero(T) == (T.rows*T.cols); // ! (sum(T(:)) != numel(T))
    bool noChangesInSea = cv::countNonZero(dT) == 0; // ! (sum(dT(:)) != 0)

    if (not everythingIsSea && not noChangesInSea){ // (sum(T(:)) != numel(T)) && (sum(dT(:)) != 0)
        std::vector<Point> contour;
        extractTheLargestCurve(dT, contour);
        removeCirclebackY(contour);
        cv::Point firstPoint,lastPoint;
        firstPoint.x = contour[0].x; firstPoint.y = 0;
        lastPoint.x = contour[contour.size()-1].x; lastPoint.y = min(contour[contour.size()].y+2,dT.rows);

        contour.insert(contour.begin(),firstPoint);
        contour.push_back(lastPoint);

        int i;
        cv::Point t0;
        for (i=0; i<contour.size(); i++){
            contour[i].x++;
            t0 = contour[i];
            contour[i].x = t0.y;
            contour[i].y = t0.x;
        }

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
    xy_subset = Data.clone();

    float x0,y0,x1,y1;
    x0 = 0; // x0
    y0 = -(a[0]*x0 + a[2])/a[1];
    x1 = originalFrameSize.width;
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
    for (i=0; i<CC.size(); i++){
        std::vector<int> pixels = CC2pixels(CC[i]);
        std::vector<int> x,y;
        ind2sub(Size(50,50),pixels, x,y);

        int xmin,ymin,xmax,ymax,ymin_original;
        xmin = *min_element(x.begin(), x.end());
        ymin = *min_element(y.begin(), y.end()); ymin_original=ymin;
        xmax = *max_element(x.begin(), x.end());
        ymax = *max_element(y.begin(), y.end());

        int width, height;
        xmax++;ymax++;
        width = xmax-xmin;
        height = ymax-ymin;

        xmin*=Tt.at<float>(0,0);
        ymin*=Tt.at<float>(1,1);
        width*=Tt.at<float>(0,0);
        height*=Tt.at<float>(1,1);
        std::vector <float> boundingBox;

        boundingBox.push_back((float)xmin);
        boundingBox.push_back((float)ymin);
        boundingBox.push_back((float)width);
        boundingBox.push_back((float)height);

        float area = width*height;
        int loc = 0;
        while(y[loc]!=ymin_original) loc++;
        xmin = x[loc]*(int)Tt.at<float>(1,1);

        cv::Mat temp;
        temp = xy_subset.row(0).clone();
        temp.convertTo(temp,CV_32F);
        std::vector <float> temp2;
        int j;
        for(j=0;j<temp.cols;j++){
            temp2.push_back(abs(temp.at<float>(j)-xmin));
        }
        float temp2_min;
        temp2_min = *min_element(temp2.begin(), temp2.end());
        loc=0;
        while(temp2[loc]!=temp2_min) loc++;

        cv::Mat boundary = xy_subset.col(loc);

        if (boundary.at<float>(1)>ymin){
            continue;
        }

        // objekt je prestal test, dodajmo ga v vektor
        object thisObject;
        thisObject.bounding_box = boundingBox;
        thisObject.area = area;

        objects.push_back(thisObject);
    }

    suppressDetections(objects, suppressedObjects);

    return;
}

void keepLargestBlob(cv::Mat &in, cv::Mat &out){
    // TODO: pomembno!!! Ne deluje pravilno ce ima najvecji blob 'lukno':

    // ne operiramo na in, ker smo ga nekako ponesreči spreminjali :/
    // ne operiramo na out, ker vmes izhodna matrika prevzame drugačno velikost kot jo ima out, ki je bil deklariran in inicializiran že zunaj - pred klicem funkcije
    // TODO: kaj če out ne bi bil inicializiran že pred klicem te funckije?
    cv::Mat src; // to bo kopiju in-a
    cv::Mat dst(in.rows+2,in.cols+2,CV_8UC1,Scalar::all(0)); // To bo 'kopija' out-a
    src = in.clone();
    // vhodni matriki moramo dodati okrog črno obrobo (en piksel debelo)
    // obroba je potrebna da iskanje obrobe največjega objekta deluje pravilno
    cv::Mat empty_col, empty_row;
    empty_col = cv::Mat::zeros(src.rows, 1, src.type());
    empty_row = cv::Mat::zeros(1, src.rows+2, src.type());
    cv::hconcat(empty_col,src,src);
    cv::hconcat(src,empty_col,src);
    cv::vconcat(empty_row,src,src);
    cv::vconcat(src,empty_row,src);

    // src is binary image (CV_8U) - 0 is background, 255 are objects
    // dst is binary image (CV_8U) of only the largest blob in src
    if (src.type()!=CV_8U) {
        std::cerr << "src must be a binary image! src.type() must be CV_8U!";
    };
    if (dst.type()!=CV_8U) {
        std::cerr << "dst must be a binary image! dst.type() must be CV_8U!";
    };

    // najdi obrise in najdi največjega:
    vector<vector<Point> > contours; // Vector for storing contours
    vector<Vec4i> hierarchy; // required for function findContours()
    findContours( src, contours, hierarchy,CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image
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
    Scalar color( 255); // white color
    drawContours( dst, contours,largest_contour_index, color, CV_FILLED, 8, hierarchy ); // Draw the largest contour using previously stored index.

    dst = dst.mul(src); // zato da ohranimo oviro, ki je v src oznacena z niclami, v dst pa jo je lahko drawContours prerisal z enkami (je povsem oblita z morjem)
    dst = dst.rowRange(1,dst.rows-1).colRange(1,dst.cols-1).clone(); // obrezemo dodane stolpce (prvi in zadnji) in vrstice (prva in zadnja)

    assert(dst.rows == in.rows); // koncna velikost mora biti enaka zacetni!
    assert(dst.cols == in.cols);

    out = dst.clone();

    assert(out.rows == in.rows); // koncna velikost mora biti enaka zacetni!
    assert(out.cols == in.cols);

    return;
}

void extractTheLargestCurve(const cv::Mat &in, std::vector<cv::Point> &points){
    cv::Mat src = in.clone();
    // src is binary image matrix (CV_8U)
    // points is output vector of points representing the largest curve
    if (src.type()!=CV_8U){
        std::cerr << "src.type() must be CV_8U! source image must be binary!" << std::endl;
    }

//    cv::Mat empty_col, empty_row;
//    empty_col = cv::Mat::zeros(src.rows, 1, src.type());
//    empty_row = cv::Mat::zeros(1, src.rows+2, src.type());
//    cv::hconcat(empty_col,src,src);
//    cv::hconcat(src,empty_col,src);
//    cv::vconcat(empty_row,src,src);
//    cv::vconcat(src,empty_row,src);

    // find all curves
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    cv::findContours( src, contours, hierarchy, CV_RETR_LIST , CV_CHAIN_APPROX_NONE );


    // TODO: kaj naj se zgodi ko NI črt?
    if (contours.size() == 0){
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
    return;
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
    cv::Mat src;
    std::vector<cv::Mat> out;
    src = bw.clone();
    // vhodni matriki moramo dodati okrog črno obrobo (en piksel debelo)
    // obroba je potrebna da iskanje obrobe največjega objekta deluje pravilno
    cv::Mat empty_col, empty_row;
    empty_col = cv::Mat::zeros(src.rows, 1, src.type());
    empty_row = cv::Mat::zeros(1, src.rows+2, src.type());
    cv::hconcat(empty_col,src,src);
    cv::hconcat(src,empty_col,src);
    cv::vconcat(empty_row,src,src);
    cv::vconcat(src,empty_row,src);
    // src is binary image (CV_8U) - 0 is background, 255 are objects
    // dst is binary image (CV_8U) of only the largest blob in src
    if (src.type()!=CV_8U) {
        std::cerr << "src must be a binary image! src.type() must be CV_8U!";
    };

    // najdi obrise in najdi največjega:
    vector<vector<Point> > contours; // Vector for storing contours
    vector<Vec4i> hierarchy; // required for function findContours()
    findContours( src, contours, hierarchy,CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image

    int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        cv::Mat currentBlob(src.rows,src.cols,CV_8U,Scalar(0));
        Scalar color( 255, 255, 255 );
        drawContours(currentBlob,contours,idx,color, CV_FILLED, 8, hierarchy);
        currentBlob = currentBlob.rowRange(1,currentBlob.rows-1).colRange(1,currentBlob.cols-1).clone(); // obrezemo dodane stolpce (prvi in zadnji) in vrstice (prva in zadnja)
        out.push_back(currentBlob.clone());
        assert(out[idx].rows == bw.rows); // koncna velikost mora biti enaka zacetni!
        assert(out[idx].cols == bw.cols);
    }

    return out;
}

void suppressDetections(const std::vector<object>& originalObjects, std::vector<object> &suppressed){
    // suppressDetections.m:1
    // function objs_out = suppressDetections(objs)

    cv::Mat bbs;
    if (originalObjects.size()>=2){
        std::vector <std::vector <int> > selected;
        mergeByProximity(bbs, originalObjects, selected);
        pruneobjs(bbs, selected, suppressed);
    }
    else{ // there is nothing to suppress
        suppressed = originalObjects;
    }
    return;
}

void pruneobjs(const cv::Mat &bbs, std::vector<std::vector <int> > selected, std::vector<object>& suppressedObjects ){
    // bbs mora biti matrika bounding box vektorjev - vsak bounding box je svoja vrstica
    // originalObjects mora biti vektor originalnih objektov
    // selected mora biti vektor z indeksi izbranih

    int i,j;
    for (i=0; i<selected.size(); i++){
        object current;

        // iz bbs matrike poberemo vrstico in jo shranimo kot vektor v current.bounding_box
        for (j=0; j<bbs.cols; j++){
            current.bounding_box.push_back((int) bbs.row(i).at<float>(j));
        }

        current.area = current.bounding_box[2]*current.bounding_box[3];

        // PARTS (starši) niso dodani v objekt!!!
        suppressedObjects.push_back(current);
    }
}

void mergeByProximity(cv::Mat& bbs_out, std::vector<object> objects, std::vector<std::vector <int> >& selected_out){
    int nObjects_in = (int)objects.size();
    std::vector <float> areas;
    int i;
    // vektor areas nafilamo z areas iz objektov
    for (i=0; i<nObjects_in; i++){
        areas.push_back(objects[i].area);
    }

    std::vector <object> orderedObjects; // urejeni od največje površine do najmanjše
    std::vector <int> order; // kako smo jih premešali?
    // order TODO: preveri a to dobro deluje?
    float minVal,maxVal; int minIdx,maxIdx;
    int j;
    for (j=0; j<nObjects_in; j++) {
        myMinMaxValIdx(areas,minVal,maxVal,minIdx,maxIdx);
        orderedObjects.push_back(objects[maxIdx]);
        order.push_back(maxIdx);
        areas[maxIdx] = minVal/2;
    }

    // iz objects[___].bounding_box v matriko bounding boxov (bbs)
    // Vsaka vrstica naj bo bounding box enega objekta
    cv::Mat bbs((int) orderedObjects.size(), (int) orderedObjects[0].bounding_box.size(),CV_32F);
    for (i=0; i<orderedObjects.size(); i++){
        bbs.row(i) = cv::Mat(orderedObjects[i].bounding_box, true).t();
    }

    // Mu = bbs(:,1:2) + (bbs(:,3:4)/2);
    cv::Mat Mu(nObjects_in,2,CV_32F);
    Mu = bbs.colRange(0,2)+bbs.colRange(2,4)/2;

    // box_sizes = sum(bbs(:,3:4),2)/2;
    cv::Mat box_sizes;
    reduce(bbs.colRange(2,4),box_sizes,REDUCE_TO_COL,CV_REDUCE_SUM);
    box_sizes /= 2;

    // Covs = (bbs(:,3:4)*1+5).^2;
    cv::Mat Covs;
    Covs = (bbs.colRange(2,4)+5);
    Covs = Covs.mul(Covs);
    //std::vector <int> selected_out;
    float mindist = 1;
    int counter = 1;

    while (true) {
        std::vector <float> ratios;
        cv::Mat C1,C2,C;
        Covs.row(0).copyTo(C1);

        for (i=0; i<bbs.rows; i++){
            Covs.row(i).copyTo(C2);
            C = C1+C2;
            cv::Mat temp, temp2, temp3;
            temp = Mu.row(0)-Mu.row(i); // Mu(1,:)-Mu(i,:)
            temp = temp.mul(temp); // (Mu(1,:)-Mu(i,:)).^2
            cv::divide(temp,C,temp2,1,CV_32F); // ((Mu(1,:)-Mu(i,:)).^2)./C
            cv::reduce(temp2,temp3,REDUCE_TO_COL,CV_REDUCE_SUM,CV_32F); // sum(((Mu(1,:)-Mu(i,:)).^2)./C)
            ratios.push_back(sqrt(temp3.at<float>(0))); // ratios(i) = sqrt(sum(((Mu(1,:)-Mu(i,:)).^2)./C)
        }

        // id_remove = (ratios <= mindist)
        std::vector <bool> id_remove; int n=0;
        for (i=0; i<ratios.size(); i++){
            id_remove.push_back(ratios[i]<=mindist);
//            id_remove.push_back(false);

            n+=(int)ratios[i]<=mindist;
        }

        cv::Mat bbs_temp;
        if (n) {
            suppress_detections(bbs, Mu, id_remove, bbs_temp);
        }
        else{
            bbs_temp = bbs.clone();
        }

        if (!bbs_out.empty())
            cv::vconcat(bbs_out,bbs_temp,bbs_out);
        else
            bbs_out = bbs_temp.clone();


        // selected_out(counter).idx = ordr(id_remove);
        std::vector <int> temp;
        for (i=0; i<order.size(); i++){
            if (id_remove[i]){
                temp.push_back(order[i]);
            }
        }
        selected_out.push_back(temp);

        removeRows(bbs,bbs,id_remove); // bbs(id_remove,:) = [];
        removeRows(Mu,Mu,id_remove); // Mu(id_remove,:) = [];
        removeVectorElementsInt(order, order, id_remove); // ordr(id_remove,:) = []; TODO: zamenjaj s template
        removeRows(Covs, Covs, id_remove);
        removeRows(box_sizes,box_sizes,id_remove);

        if(Mu.rows == 0){ // if (isempty(Mu))
            break;
        }

        counter++;
    }
    return;
}

void suppress_detections(cv::Mat bbs_in, cv::Mat Mu_in, std::vector<bool> selected, cv::Mat &bbs_out){
    // bbs = bbs_in(selected,:); Mu = Mu_in(selected,:);
    int i, n = 0, n_selected = 0;
    for (i=0;i<selected.size();i++) n_selected+=(int)selected[i];
    cv::Mat bbs(n_selected,bbs_in.cols,bbs_in.type()), Mu(n_selected,Mu_in.cols,bbs_in.type());
    for (i=0; i<selected.size(); i++){
        if(selected[i]){
            bbs_in.row(i).copyTo(bbs.row(n));
            Mu_in.row(i).copyTo(Mu.row(n));
            n++;
        }
    }

    cv::Mat minX, maxX, minY, maxY;
    minX = Mu.col(0) - bbs.col(2) / 2;
    maxX = Mu.col(0) + bbs.col(2) / 2;
    minY = Mu.col(1) - bbs.col(3) / 2;
    maxY = Mu.col(1) + bbs.col(3) / 2;

    cv::reduce(minX, minX, REDUCE_TO_ROW, CV_REDUCE_MIN, CV_32F);
    cv::reduce(minY, minY, REDUCE_TO_ROW, CV_REDUCE_MIN, CV_32F);
    cv::reduce(maxX, maxX, REDUCE_TO_ROW, CV_REDUCE_MAX, CV_32F);
    cv::reduce(maxY, maxY, REDUCE_TO_ROW, CV_REDUCE_MAX, CV_32F);

    std::vector<float> bbs_out_vector;
    bbs_out_vector.push_back(minX.at<float>(0));
    bbs_out_vector.push_back(minY.at<float>(0));
    bbs_out_vector.push_back(maxX.at<float>(0) - minX.at<float>(0));
    bbs_out_vector.push_back(maxY.at<float>(0) - minY.at<float>(0));

    bbs_out = cv::Mat(bbs_out_vector, CV_32F).t(); // vector to matrix
}

std::vector<int> CC2pixels(cv::Mat CC){
    cv::Mat formated;
    CC.convertTo(formated, CV_64F);

    std::vector<int> out;
    int r,c;
    for (c=0; c<formated.cols; c++){
        for (r=0; r<formated.rows; r++){
            if(formated.row(r).col(c).at<double>(0)){
                out.push_back(r + c*formated.rows);
            }
        }
    }

    return out;
}

void ind2sub(cv::Size sz, std::vector<int> ind, std::vector<int> &x, std::vector<int> &y){
    int i,r,c;
    for (i=0;i<ind.size();i++){
        c = ind[i]/(int)sz.width; // celoštevilsko deljenje ker sta oba integerja!
        r = ind[i] % sz.height;
        assert(r == (ind[i]-c*sz.height));
        x.push_back(c); y.push_back(r);
    }
}