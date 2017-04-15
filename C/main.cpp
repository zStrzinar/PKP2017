//
// Created by ziga on 8.4.2017.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "utility.h"

using namespace cv;
using namespace std;

int main (int argc, char ** argv){


    // -------------------------- Obdelava vhodnih argumentov ----------------------------------------------------------
    // Najprej samo obdelava vhodnih argumentov
    //  cilj obdelave je, da imamo na koncu inputPath, outputPath, inputFormat in outputFormat
    //  obdelava tudi lahko kliče pomoč (-h)
    std::string inputPath, outputPath;
    std::string inputFormat, outputFormat;
    switch (argc) {
        case 1:	{
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
        case 3: {
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

    std::cout << "Input file path: " << inputPath << std::endl;
    //std::cout << "Input file format: " << inputFormat << std::endl;

    std::cout << "Output file path: " << outputPath << std::endl;
    //std::cout << "Output file format: " << outputFormat << std::endl;

    // ---------------------------------- Odpiranje videa in definiranje nekaterih nastavitev --------------------------

    std::cout << "Opening video file" << std::endl;
    cv::VideoCapture inputVideo(inputPath);
    if (!inputVideo.isOpened())
        std::cerr << "Failed to open video file! Aborting" << endl;
    else
        std::cout << "Video file successfully opened" << endl;

    // Settings
    bool use_prior_on_mixture = true; // MATLAB: example.m:39 % detector constructor
    bool use_uniform_component = true; // MATLAB: example.m:39 % detector constructor
    Colorspace colorSpace = YCRCB; // MATLAB: example.m:39 % detector constructor
    double min_lik_delta = 1e-2; // MATLAB: example.m:39 % detector constructor
    int maxEMsteps = 10; // MATLAB: example.m:39 % detector constructor
    cv::Mat PI_i = Mat(); // MATLAB: example.m:39 % detector constructor
    std::vector<cv::Mat> mix_Mu, mix_Cov, mix_w, static_prec;
    loadPriorModelFromDisk(colorSpace, mix_Mu, mix_Cov, mix_w, static_prec); // hardcoded values
    int velikost[] = {50,50};
    std::vector<int> em_image_size(velikost,velikost+2);

    // Spacial data mora biti: v prvi vrstici 1:50 potem pa spet 1:50 in spet 1:50  petdesetkrat...
    // V drugi vrstici pa mora biti najprej 50x 1 potem 50x 2 potem 50x 3 in tako naprej do 50.
    cv::Mat prvaVrstica_kratko(1,em_image_size[0],CV_64F);
    int i;
    for (i=0;i<em_image_size[0];i++){
        prvaVrstica_kratko.at<double>(i)=i+1.0;
    }
    // prvaVrstica_kratko je zdaj [1,2,3,...,49,50]. Zdaj moram še 50x ponovit
    cv::Mat prvaVrstica = repeat(prvaVrstica_kratko,1,em_image_size[1]); // ponovitve
    // druga vrstica: najprej naredim stolpično matriko z vrednostmi [1,2,...,50]'.
    // Potem jo razširim iz enega stolpca v 50 stolpcev [1,1,1,...,1,1;2,2,2...,2,2;...;50,50,...,50]
    // Potem pa reshapeam iz 50 vrstic v 1 vrstico [1,1,...,1,1,2,2,...,2,2,....,50,5,...,50,50]
    cv::Mat drugaVrstica_stolpicni(em_image_size[1],1,CV_64F);
    for (i=0; i<em_image_size[1]; i++){
        drugaVrstica_stolpicni.at<double>(i)=i+1.0;
    } // stolpicni zapolnjen
    cv::Mat drugaVrstica_matrika = repeat(drugaVrstica_stolpicni,1,em_image_size[0]); // stopiram stolpec 50x
    cv::Mat drugaVrstica = drugaVrstica_matrika.reshape(0,1).clone();
    // zlepim skupaj prvo in drugo vrstico
    cv::Mat spatial_data;//(2,em_image_size[0]*em_image_size[1],CV_64F);
    cv::vconcat(prvaVrstica,drugaVrstica,spatial_data);
    spatial_data = spatial_data.clone();
    // ----------------------- nastavitve dokončane --------------------------------------------------------------------

    std::cout << "Detector initialized" << std::endl;

    std::cout << "Beginning frame-by-frame algorithm" << std::endl;
    int frame_number;
    cv::Mat frame_original, frame_resized, frame_colorspace;
    cv::Mat color_data_rows[3];
    cv::Size resized_size(em_image_size[0],em_image_size[1]);
    cv::Size original_size;

    for(frame_number = 1; frame_number<=inputVideo.get(CV_CAP_PROP_FRAME_COUNT); frame_number++){
        inputVideo.set(CV_CAP_PROP_POS_FRAMES, frame_number-1);
        inputVideo >> frame_original;

        original_size = frame_original.size();

        cv::resize(frame_original,frame_resized,resized_size,0,0,CV_INTER_LINEAR);
        switch (colorSpace){
            case YCRCB:{
                cv::cvtColor(frame_resized, frame_colorspace, CV_BGR2YCrCb);
                break;
            }
            default:{
                std::cerr<<"Unsupported colorspace!"<<std::endl;
            }
        }

        cv::split(frame_colorspace,color_data_rows); // color_date_rows[0] je zdaj prvi kanal frame_colorspace
        color_data_rows[0] = color_data_rows[0].reshape(0,1).clone(); // color_data_rows[0] rata vrstična matrika. Prvih 50 vrednosti je iz prve vrstice originala, naslednih 50 je iz naslednje vrstice originala itd.
        color_data_rows[1] = color_data_rows[1].reshape(0,1).clone();
        color_data_rows[2] = color_data_rows[2].reshape(0,1).clone();

        cv::Mat color_data;//(3,em_image_size[0]*em_image_size[1],CV_64F);
        cv::vconcat(color_data_rows,3,color_data); // zlepim rows v color_data
        color_data.convertTo(color_data,CV_64F);
        cv::Mat dataEM;//(5,em_image_size[0]*em_image_size[1],CV_64F);
        cv::vconcat(spatial_data,color_data,dataEM); // Zlepim skupaj color_data in spatial_data

        cv::Mat current_Mu[3], current_Cov[3], current_region;

//        std::cout << dataEM << std::endl;

        double current_w[3];
        if (frame_number==1){
            float df[] = {0,0.3,0.5,1};
            std::vector <float> vertical_ratio(df, df+sizeof(df)/sizeof(float) );
            std::transform(vertical_ratio.begin(), vertical_ratio.end(), vertical_ratio.begin(),
                           std::bind1st(std::multiplies<float>(),dataEM.cols));
            int k;
            for (k=0; k<=2; k++){
                current_region = dataEM.colRange((int)vertical_ratio[k],(int)vertical_ratio[k+1]);
                current_w[k]=1/3;
                cv::calcCovarMatrix(current_region, current_Cov[k], current_Mu[k], CV_COVAR_NORMAL|CV_COVAR_COLS);
                current_Cov[k] = current_Cov[k] / (current_region.cols - 1);
            }
            std::cout << "Covariance, region 1:" << std::endl << current_Cov[0] << std::endl;
            std::cout << "Covariance, region 2:" << std::endl << current_Cov[1] << std::endl;
            std::cout << "Covariance, region 3:" << std::endl << current_Cov[2] << std::endl;

            std::cout << "Mean, region 1:" << std::endl << current_Mu[0] << std::endl;
            std::cout << "Mean, region 2:" << std::endl << current_Mu[1] << std::endl;
            std::cout << "Mean, region 3:" << std::endl << current_Mu[2] << std::endl;

        }
        else{
            // TODO: detect_edge_of_sea_simplified.m:93
        }



        std::cout << "Frame " << frame_number << " done" << std::endl;
    }
    return 0;
}

