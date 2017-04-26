//
// Created by ziga on 8.4.2017.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "utility.h"
#include "objectDetection.h"

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
    long double eps = 2.2204e-16;
    bool use_prior_on_mixture = true; // MATLAB: example.m:39 % detector constructor
    bool use_uniform_component = true; // MATLAB: example.m:39 % detector constructor
    Colorspace colorSpace = YCRCB; // MATLAB: example.m:39 % detector constructor
//    double min_lik_delta = 1e-2; // MATLAB: example.m:39 % detector constructor
    int maxEMsteps = 10; // MATLAB: example.m:39 % detector constructor
    cv::Mat PI_i = cv::Mat(); // MATLAB: example.m:39 % detector constructor
    std::vector<cv::Mat> current_mix_Mu, current_mix_Cov, current_mix_W, current_mix_Prec;
    std::vector<cv::Mat> prior_mix_Mu, prior_mix_Cov, prior_mix_W, prior_mix_Prec;
    loadPriorModelFromDisk(colorSpace, prior_mix_Mu, prior_mix_Cov, prior_mix_W, prior_mix_Prec); // hardcoded values
//    int velikost[] = {50,50};
//    std::vector<int> em_image_size(velikost,velikost+2);
    cv::Size em_image_size(50,50);
//    cv::Mat em_image_size_mat = cv::Mat(2,1,CV_32S, velikost).clone();
    // Spacial data mora biti: v prvi vrstici 1:50 potem pa spet 1:50 in spet 1:50  petdesetkrat...
    // V drugi vrstici pa mora biti najprej 50x 1 potem 50x 2 potem 50x 3 in tako naprej do 50.
    cv::Mat prvaVrstica_kratko(1,em_image_size.width,CV_64F);
    int i;
    for (i=0;i<em_image_size.width;i++){
        prvaVrstica_kratko.at<double>(i)=i+1.0;
    }
    // prvaVrstica_kratko je zdaj [1,2,3,...,49,50]. Zdaj moram še 50x ponovit
    cv::Mat prvaVrstica = repeat(prvaVrstica_kratko,1,em_image_size.height); // ponovitve
    // druga vrstica: najprej naredim stolpično matriko z vrednostmi [1,2,...,50]'.
    // Potem jo razširim iz enega stolpca v 50 stolpcev [1,1,1,...,1,1;2,2,2...,2,2;...;50,50,...,50]
    // Potem pa reshapeam iz 50 vrstic v 1 vrstico [1,1,...,1,1,2,2,...,2,2,....,50,5,...,50,50]
    cv::Mat drugaVrstica_stolpicni(em_image_size.height,1,CV_64F);
    for (i=0; i<em_image_size.height; i++){
        drugaVrstica_stolpicni.at<double>(i)=i+1.0;
    } // stolpicni zapolnjen
    cv::Mat drugaVrstica_matrika = repeat(drugaVrstica_stolpicni,1,em_image_size.width); // stopiram stolpec 50x
    cv::Mat drugaVrstica = drugaVrstica_matrika.reshape(0,1).clone();
    // zlepim skupaj prvo in drugo vrstico
    cv::Mat spatial_data;//(2,em_image_size[0]*em_image_size[1],CV_64F);
    cv::vconcat(prvaVrstica,drugaVrstica,spatial_data);
    spatial_data = spatial_data.clone();
    std::string type_of_em = "em_seg"; // em_basic em_seg em_basic_no_smooth

    // ----------------------- nastavitve dokončane --------------------------------------------------------------------

    std::cout << "Detector initialized" << std::endl;

    std::cout << "Beginning frame-by-frame algorithm" << std::endl;
    int frame_number;
    cv::Mat frame_original, frame_resized, frame_colorspace;
    cv::Mat color_data_rows[3];
    cv::Size resized_size(em_image_size.width*4,em_image_size.height*4);
    cv::Size original_size;
    cv::namedWindow("Moje okno",WINDOW_AUTOSIZE);
    for(frame_number = 1; frame_number<=inputVideo.get(CV_CAP_PROP_FRAME_COUNT); frame_number++){
        inputVideo.set(CV_CAP_PROP_POS_FRAMES, frame_number-1);
        inputVideo >> frame_original;

        frame_original = imread("00001.jpg", CV_LOAD_IMAGE_COLOR); // TODO: to je samo za debug!
        original_size = frame_original.size();

        cv::resize(frame_original,frame_resized,resized_size,0,0,INTER_LINEAR);

        frame_resized = imread("temp.jpg", CV_LOAD_IMAGE_COLOR); // TODO: to je samo za debugiranje!


        switch (colorSpace){
            case YCRCB:{
                cv::cvtColor(frame_resized, frame_colorspace, CV_BGR2YCrCb);
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
                    temp = color_data_rows[1].clone();
                    color_data_rows[0] = color_data_rows[0].reshape(0, 1).clone(); // color_data_rows[0] rata vrstična matrika. Prvih 50 vrednosti je iz prve vrstice originala, naslednih 50 je iz naslednje vrstice originala itd.
                    color_data_rows[1] = color_data_rows[2].reshape(0, 1).clone();
                    color_data_rows[2] = temp.reshape(0, 1).clone();
            }
        else{
            color_data_rows[0] = color_data_rows[0].reshape(0, 1).clone(); // color_data_rows[0] rata vrstična matrika. Prvih 50 vrednosti je iz prve vrstice originala, naslednih 50 je iz naslednje vrstice originala itd.
            color_data_rows[1] = color_data_rows[1].reshape(0, 1).clone();
            color_data_rows[2] = color_data_rows[2].reshape(0, 1).clone();
        }


        cv::Mat color_data;//(3,em_image_size[0]*em_image_size[1],CV_64F);
        cv::vconcat(color_data_rows,3,color_data); // zlepim rows v color_data
        color_data.convertTo(color_data,CV_64F);
        cv::Mat dataEM;//(5,em_image_size[0]*em_image_size[1],CV_64F);
        cv::vconcat(spatial_data,color_data,dataEM); // Zlepim skupaj color_data in spatial_data
        std::cout << std::endl << dataEM << std::endl;
        cv::Mat current_Mu[3], current_Cov[3], current_region;
//        cv::Mat current_region;
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
                current_w[k]=1.0/3.0;
                cv::calcCovarMatrix(current_region, current_Cov[k], current_Mu[k], CV_COVAR_NORMAL|CV_COVAR_COLS);
                current_Cov[k] = current_Cov[k] / (current_region.cols - 1);
                current_mix_W.insert(current_mix_W.end(), cv::Mat(1,1,CV_64F, Scalar(1.0/3.0)));
                current_mix_Cov.insert(current_mix_Cov.end(), current_Cov[k]);
                current_mix_Mu.insert(current_mix_Mu.end(), current_Mu[k]);
            }
            /*std::cout << "Covariance, region 1:" << std::endl << current_mix_Cov[0] << std::endl;
            std::cout << "Covariance, region 2:" << std::endl << current_mix_Cov[1] << std::endl;
            std::cout << "Covariance, region 3:" << std::endl << current_mix_Cov[2] << std::endl;

            std::cout << "Mean, region 1:" << std::endl << current_mix_Mu[0] << std::endl;
            std::cout << "Mean, region 2:" << std::endl << current_mix_Mu[1] << std::endl;
            std::cout << "Mean, region 3:" << std::endl << current_mix_Mu[2] << std::endl;*/
        }
        else{
            // TODO: detect_edge_of_sea_simplified.m:93
        }

        cv::Mat Q_sum_large, mix_PI_i;
        std::cout << "colorSpace: " << colorSpace << std::endl // TODO: samo za debugiranje
                  << "em_image size: " << em_image_size << std::endl
                  << "use_uniform_component: " << use_uniform_component << std::endl
                  << "type_of_em: " << type_of_em << std::endl
                  << "maxEMsteps: " << maxEMsteps << std::endl
                  << "current_mix_W[0]: " << current_mix_W[0] << std::endl
                  << "current_mix_W[1]: " << current_mix_W[1] << std::endl
                  << "current_mix_W[2]: " << current_mix_W[2] << std::endl
                  << "current_mix_Mu[0]: " << current_mix_Mu[0] << std::endl
                  << "current_mix_Mu[1]: " << current_mix_Mu[1] << std::endl
                  << "current_mix_Mu[2]: " << current_mix_Mu[2] << std::endl
                  << "current_mix_Cov[0]: " << current_mix_Cov[0] << std::endl
                  << "current_mix_Cov[1]: " << current_mix_Cov[1] << std::endl
                  << "current_mix_Cov[2]: " << current_mix_Cov[2] << std::endl
                  << "prior_mix_W[0]: " << prior_mix_W[0] << std::endl
                  << "prior_mix_W[1]: " << prior_mix_W[1] << std::endl
                  << "prior_mix_W[2]: " << prior_mix_W[2] << std::endl
                  << "prior_mix_Mu[0]: " << prior_mix_Mu[0] << std::endl
                  << "prior_mix_Mu[1]: " << prior_mix_Mu[1] << std::endl
                  << "prior_mix_Mu[2]: " << prior_mix_Mu[2] << std::endl
                  << "prior_mix_Cov[0]: " << prior_mix_Cov[0] << std::endl
                  << "prior_mix_Cov[1]: " << prior_mix_Cov[1] << std::endl
                  << "prior_mix_Cov[2]: " << prior_mix_Cov[2] << std::endl
                  << "prior_mix_Prec[0]: " << prior_mix_Prec[0] << std::endl
                  << "prior_mix_Prec[1]: " << prior_mix_Prec[1] << std::endl
                  << "prior_mix_Prec[2]: " << prior_mix_Prec[2] << std::endl; // TODO: samo za debugiranje
        run_SSM(colorSpace, em_image_size, use_uniform_component, type_of_em,
                maxEMsteps, current_mix_W, PI_i, dataEM, current_mix_Mu, current_mix_Cov, prior_mix_Mu,
                prior_mix_Prec, use_prior_on_mixture, eps, Q_sum_large, mix_PI_i);

        std::vector <cv::Mat>PI_i_channels;
        cv::split(mix_PI_i, PI_i_channels);
        std::cout << "PI_i(:,:,1)=" << PI_i_channels[0] << std::endl; // TODO: samo za debugiranje
        std::cout << "PI_i(:,:,2)=" << PI_i_channels[1] << std::endl; // TODO: samo za debugiranje
        std::cout << "PI_i(:,:,3)=" << PI_i_channels[2] << std::endl; // TODO: samo za debugiranje
        std::cout << "PI_i(:,:,4)=" << PI_i_channels[3] << std::endl; // TODO: samo za debugiranje
        std::cout << "Frame " << frame_number << " done" << std::endl;

        getEdgeAndObjectNoScaling(Q_sum_large, original_size);
    }
    return 0;
}