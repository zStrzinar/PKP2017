//
// Created by ziga on 8.4.2017.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include "utility.h"
#include "objectDetection.h"
#include "showResults.h"
#include "zStrzinar.h"
#include "SSM.h"

using namespace cv;
using namespace std;

void printHelp();

int main (int argc, char ** argv){
    // -------------------------- Obdelava vhodnih argumentov ----------------------------------------------------------
    // Najprej samo obdelava vhodnih argumentov
    //  cilj obdelave je, da imamo na koncu inputPath, outputPath, inputFormat in outputFormat
    //  obdelava tudi lahko kliče pomoč (-h)
    std::string inputPath, outputPath;
    std::string inputFormat, outputFormat;
    switch (argc) {
        case 1:	{ // klic programa brez dodatnih argumentov
            printHelp();
            return 1;
        }
        case 2:	{
            if (strcmp("-h", argv[1]) == 0) {
                printHelp();
                return 1;
            }
            else {
                inputPath = argv[1];
                std::size_t idx = inputPath.find_last_of('.');
                outputPath = inputPath.substr(0, idx);
                inputFormat = inputPath.substr(idx + 1);
                outputPath.append("_segmented.");
                outputPath.append(inputFormat);
                outputFormat = inputFormat;
                break;
            }
        }
        case 3: { // podana tako vhodna kot izhodna datoteka
            inputPath = argv[1];
            outputPath = argv[2];
            std::size_t idx = inputPath.find_last_of('.');
            inputFormat = inputPath.substr(idx + 1);
            idx = outputPath.find_last_of('.');
            outputFormat = outputPath.substr(idx + 1);
            break;
        }
        default: {
            printHelp();
            return 1;
        }
    }
    std::cout << "Input file path: " << inputPath << std::endl << "Output file path: " << outputPath << std::endl;

    // ---------------------------------- Odpiranje videa in definiranje nekaterih nastavitev --------------------------

    std::cout << "Opening video file" << std::endl;
    cv::VideoCapture inputVideo(inputPath);
    if (!inputVideo.isOpened())
        std::cerr << "Failed to open video file! Aborting" << endl;
    else
        std::cout << "Video file successfully opened" << endl;

    cv::VideoWriter outputVideo;
    int ex = static_cast<int>(inputVideo.get(CV_CAP_PROP_FOURCC));
    Size S = Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
    outputVideo.open(outputPath,ex, inputVideo.get(CV_CAP_PROP_FPS), S, true);
    // Settings
    long double eps = 2.2204e-16;
    bool use_prior_on_mixture = true; // MATLAB: example.m:39 % detector constructor
    bool use_uniform_component = true; // MATLAB: example.m:39 % detector constructor
    Colorspace colorSpace = YCRCB; // MATLAB: example.m:39 % detector constructor
    int maxEMsteps = 10; // MATLAB: example.m:39 % detector constructor
    cv::Mat PI_i = cv::Mat(); // MATLAB: example.m:39 % detector constructor
    std::vector<cv::Mat> current_mix_Mu, current_mix_Cov, current_mix_W, current_mix_Prec;
    std::vector<cv::Mat> prior_mix_Mu, prior_mix_Cov, prior_mix_W, prior_mix_Prec;
    cv::Size em_image_size(50,50);
    cv::Mat spatial_data; getSpacialData(em_image_size, spatial_data);
    std::string type_of_em = "em_seg"; // em_basic em_seg em_basic_no_smooth
    std::cout << "Detector initialized" << std::endl;
    // -----------------------------------------------------------------------------------------------------------------

    std::cout << "Beginning frame-by-frame algorithm" << std::endl;
    int frame_number;
    cv::Mat frame_original, frame_resized, frame_colorspace;
    cv::Mat color_data_rows[3];
    cv::Size resized_size(em_image_size.width,em_image_size.height);
    cv::Size original_size;
    for(frame_number = 1; frame_number<=inputVideo.get(CV_CAP_PROP_FRAME_COUNT); frame_number++){

        inputVideo.set(CV_CAP_PROP_POS_FRAMES, frame_number-1);
        inputVideo >> frame_original;

        char currentFile[17];

        original_size = frame_original.size();

        cv::resize(frame_original,frame_resized,resized_size,0.5,0.5,INTER_LINEAR);

        switch (colorSpace){
            case YCRCB:{
                myBGR2YCrCb(frame_resized, frame_colorspace);
                // YCrCb2YCbCr (openCV2matlab=
                std::vector<cv::Mat> YCrCb, YCbCr;
                cv::split(frame_colorspace, YCrCb);
                YCbCr.push_back(YCrCb[0]);
                YCbCr.push_back(YCrCb[2]);
                YCbCr.push_back(YCrCb[1]);
                merge(YCbCr,frame_colorspace);

                break;
            }
            case HSV:break;
            case RGB:break;
            case LAB:break;
            case YCRS:break;
            case NONE:break;
            default:{
                std::cerr<<"Unsupported colorspace!"<<std::endl;
            }
        }

        cv::split(frame_colorspace,color_data_rows); // color_date_rows[0] je zdaj prvi kanal frame_colorspace
        if (colorSpace == YCRCB){
                    cv::Mat temp;
                    color_data_rows[0] = color_data_rows[0].reshape(0, 1).clone(); // color_data_rows[0] rata vrstična matrika. Prvih 50 vrednosti je iz prve vrstice originala, naslednih 50 je iz naslednje vrstice originala itd.
                    color_data_rows[1] = color_data_rows[1].reshape(0, 1).clone();
                    color_data_rows[2] = color_data_rows[2].reshape(0, 1).clone();
            }
        else{
            color_data_rows[0] = color_data_rows[0].reshape(0, 1).clone(); // color_data_rows[0] rata vrstična matrika. Prvih 50 vrednosti je iz prve vrstice originala, naslednih 50 je iz naslednje vrstice originala itd.
            color_data_rows[1] = color_data_rows[1].reshape(0, 1).clone();
            color_data_rows[2] = color_data_rows[2].reshape(0, 1).clone();
        }

        cv::Mat color_data;
        cv::vconcat(color_data_rows,3,color_data); // zlepim rows v color_data
        color_data.convertTo(color_data,CV_64F);
        cv::Mat dataEM;

        cv::vconcat(spatial_data,color_data,dataEM); // Zlepim skupaj color_data in spatial_data
        cv::Mat current_Mu[3], current_Cov[3], current_region;
        if (frame_number==1){
//        if (!((frame_number-1)%30)){
            loadPriorModelFromDisk(colorSpace, prior_mix_Mu, prior_mix_Cov, prior_mix_W, prior_mix_Prec); // hardcoded values
//            float df[] = {0,0.3,0.5,1}; // to so rocno nastavljene zacetne meje med območji
            float df[] = {0,0.2,0.6,1}; // to so rocno nastavljene zacetne meje med območji
            std::vector <float> vertical_ratio(df, df+sizeof(df)/sizeof(float) ); // to samo ustvari vektor in ga zafila z vrednostmi df[]
            std::transform(vertical_ratio.begin(), vertical_ratio.end(), vertical_ratio.begin(),
                           std::bind1st(std::multiplies<float>(),dataEM.cols)); // to pomnoži vektor z dataEM.cols
            // oboje skupaj je v matlabu: vertical_ratio = df*size(dataEM,2)

            int k;
            if (current_mix_W.size()>0) current_mix_W.clear();
            if (current_mix_Cov.size()>0) current_mix_Cov.clear();
            if (current_mix_Mu.size()>0) current_mix_Mu.clear();
            for (k=0; k<=2; k++){ // cez vse regije:
                current_region = dataEM.colRange((int)vertical_ratio[k],(int)vertical_ratio[k+1]); // v dataEM so vse lokaije kar po vrsti v vrstici
                // TODO: ali je tukaj (eno vrstico gor) pravilen range? vključenost prvega in zadnjega stolpca v primerjavi z Matlabom?

                cv::calcCovarMatrix(current_region, current_Cov[k], current_Mu[k], CV_COVAR_NORMAL|CV_COVAR_COLS|CV_COVAR_SCALE); // Kovariančna matrika in srednja vrednost
                current_Cov[k] = current_Cov[k] * current_region.cols / (current_region.cols - 1);

                // Vpis kovariančne matrike, srednje vrednosti, w v ustrezne vektorje (current_mix_...):
                current_mix_W.insert(current_mix_W.end(), cv::Mat(1,1,CV_64F, Scalar(1.0/3.0)));
                current_mix_Cov.insert(current_mix_Cov.end(), current_Cov[k]);
                current_mix_Mu.insert(current_mix_Mu.end(), current_Mu[k]);
            }
        }
        else{
            if (!(colorSpace==HSV)){
                float df[] = {0,0.2,0.2,0.4,0.6,1}; // to so rocno nastavljene zacetne meje med območji
                std::vector <float> vertical_ratio(df, df+sizeof(df)/sizeof(float) ); // to samo ustvari vektor in ga zafila z vrednostmi df[]
                std::transform(vertical_ratio.begin(), vertical_ratio.end(), vertical_ratio.begin(),
                               std::bind1st(std::multiplies<float>(),dataEM.cols)); // to pomnoži vektor z dataEM.cols
                // oboje skupaj je v matlabu: vertical_ratio = df*size(dataEM,2)
                float w_init = 0.4;
                float w_mix_a[] = {1-w_init, w_init};
                std::vector <float> w_mix(w_mix_a, w_mix_a+sizeof(w_mix_a)/sizeof(float));

                int k;
                for (k=0; k<3; k++){ // cez vse regije
                    current_region = dataEM.colRange((int)vertical_ratio[2*k], (int)vertical_ratio[2*k+1]); // 2*k je zato ker tukaj je vertical_ratio kombinacija: [zacetek,konec,zacetek,konec,zacetek,konec]
                    // TODO: ali je tukaj (eno vrstico gor) pravilen range? vključenost prvega in zadnjega stolpca v primerjavi z Matlabom?
                    cv::calcCovarMatrix(current_region, current_Cov[k], current_Mu[k], CV_COVAR_NORMAL|CV_COVAR_COLS|CV_COVAR_SCALE, current_region.type()); // Kovariančna matrika in srednja vrednost
                    current_Cov[k] = current_Cov[k] * current_region.cols / (current_region.cols - 1);

                    momentMatchPdf(current_mix_Mu[k], current_Mu[k], current_mix_Cov[k], current_Cov[k], w_mix, current_mix_Mu[k], current_mix_Cov[k], current_mix_W[k]);
                }
                float sum_w = 0;
                for (k=0; k<3; k++){ // seštevek w po vseh regijah
                    sum_w += current_mix_W[k].at<double>(0,0);
                }
                for (k=0; k<3; k++){ // normiranje
                    current_mix_W[k]/=sum_w;
                }
            }
        }

        cv::Mat Q_sum_large, mix_PI_i;

        run_SSM(colorSpace, em_image_size, use_uniform_component, type_of_em,
                maxEMsteps, current_mix_W, PI_i, dataEM, current_mix_Mu, current_mix_Cov, prior_mix_Mu,
                prior_mix_Prec, use_prior_on_mixture, eps, Q_sum_large, mix_PI_i);

        std::vector <cv::Mat>PI_i_channels;
        cv::split(mix_PI_i, PI_i_channels);

        std::vector <object> detectedObjects;
        cv::Mat xy_subset, sea_region; std::vector<object> suppressedObjects;
        getEdgeAndObjectNoScaling(Q_sum_large, original_size, detectedObjects, xy_subset, suppressedObjects, sea_region);

        // Display resultes
        cv::Mat imageToShow;
        displayEdgeAndObjects(frame_original.clone(), imageToShow, xy_subset, suppressedObjects, sea_region, current_mix_Mu, current_mix_Cov, current_mix_W, 1);

        outputVideo << imageToShow;
        std::cout << "Frame " << frame_number << " done" << std::endl;
    }
    return 0;
}

void printHelp() {
    std::string msg;
    msg = "Help\n\tUsage:\n\t\t-h ............... for help\n\t\tinput ............ specify video path\n\t\tinput output ..... specify video path and output path";
    std::cout << msg << std::endl;
}