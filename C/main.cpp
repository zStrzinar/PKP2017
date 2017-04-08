//
// Created by ziga on 8.4.2017.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

enum Colorspace { HSV, RGB, YCRCB, LAB, YCRS, NONE };
enum BsxFunOp { DIVIDE, TIMES };

void printHelp() {
    std::string msg;
    msg = "Help\n\tUsage:\n\t\t-h\t .......... for help\n\t\tpath\t ........... specify video path\n\t\tpath output\t ... specify video path and output path";
    std::cout << msg << std::endl;
}

void loadPriorModelFromDisk(Colorspace colorSpace,
                            std::vector<cv::Mat> &mix_Mu,
                            std::vector<cv::Mat> &mix_Cov,
                            std::vector<cv::Mat> &mix_w,
                            std::vector<cv::Mat> &static_prec);


void simpleTest(std::vector<cv::Mat> &io);

int main (int argc, char ** argv){
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
    loadPriorModelFromDisk(colorSpace, mix_Mu, mix_Cov, mix_w, static_prec);
    /*std::cout   << "Mix_Cov: " << std::endl
                << "page 1: " << mix_Cov[0] << std::endl
                << "page 2: " << mix_Cov[1] << std::endl
                << "page 3: " << mix_Cov[2] << std::endl
                << "Mix_mu: " << std::endl
                << "page 1: " << mix_Mu[0] << std::endl
                << "page 2: " << mix_Mu[1] << std::endl
                << "page 3: " << mix_Mu[2] << std::endl
                << "mix_w: " << std::endl
                << "page 1: " << mix_w[0] << std::endl
                << "page 2: " << mix_w[1] << std::endl
                << "page 3: " << mix_w[2] << std::endl
                << "static_prec: " << std::endl
                << "page 1: " << static_prec[0] << std::endl
                << "page 2: " << static_prec[1] << std::endl
                << "page 3: " << static_prec[2] << std::endl;*/

    return 0;
}



void loadPriorModelFromDisk(Colorspace colorSpace,
                            std::vector<cv::Mat> &mix_Mu,
                            std::vector<cv::Mat> &mix_Cov,
                            std::vector<cv::Mat> &mix_w,
                            std::vector<cv::Mat> &static_prec) {
    int length = 5;
    switch (colorSpace)
    {
        case YCRCB: {
            // Mu
            double data_mu_1[] = {25, 11.1279, 178.6491, 128.4146, 124.4288};
            double data_mu_2[] = {25, 21.6993, 84.6005, 123.6754, 127.5533};
            double data_mu_3[] = {25, 38.2647, 101.2433, 126.3468, 119.7686};
            cv::Mat mix_Mu_1 = cv::Mat(length,1,CV_64F,data_mu_1).clone();
            cv::Mat mix_Mu_2 = cv::Mat(length,1,CV_64F,data_mu_2).clone();
            cv::Mat mix_Mu_3 = cv::Mat(length,1,CV_64F,data_mu_3).clone();

            mix_Mu.insert(mix_Mu.end(),mix_Mu_1);
            mix_Mu.insert(mix_Mu.end(),mix_Mu_2);
            mix_Mu.insert(mix_Mu.end(),mix_Mu_3);

            // Cov
            double  data_cov_1[] = {  209.0705, 0, 0, 0, 0,
                                   0, 46.6508, 0, 0, 0,
                                   0, 0, 986.8448, 0, 0,
                                   0, 0, 0, 30.4178, 0,
                                   0, 0, 0, 0, 26.6255};
            double data_cov_2[] = { 206.948, 0, 0, 0, 0,
                                   0, 22.4061, 0, 0, 0,
                                   0, 0, 2636.7, 0, 0,
                                   0, 0, 0, 26.2005, 0,
                                   0, 0, 0, 0, 17.7941};
            double data_cov_3[] = { 208.7815, 0, 0, 0, 0,
                                   0, 97.6139, 0, 0, 0,
                                   0, 0, 880.9302, 0, 0,
                                   0, 0, 0, 8.7445, 0,
                                   0, 0, 0, 0, 26.1301};
            cv::Mat mix_Cov_1 = cv::Mat(length,length,CV_64F,data_cov_1).clone();
            cv::Mat mix_Cov_2 = cv::Mat(length,length,CV_64F,data_cov_2).clone();
            cv::Mat mix_Cov_3 = cv::Mat(length,length,CV_64F,data_cov_3).clone();

            mix_Cov.insert(mix_Cov.end(),mix_Cov_1);
            mix_Cov.insert(mix_Cov.end(),mix_Cov_2);
            mix_Cov.insert(mix_Cov.end(),mix_Cov_3);

            // w
            mix_w.insert(mix_w.end(),3,cv::Mat(1,1,CV_64F,Scalar(0)));

            // prec
            double  data_prec_1[] = { 0.0047831, 0, 0, 0, 0,
                                   0, 0.0214358, 0, 0, 0,
                                   0, 0, 0.0010133, 0, 0,
                                   0, 0, 0, 0.0328755, 0,
                                   0, 0, 0, 0, 0.037558};
            double data_prec_2[] = {0.0048321, 0, 0, 0, 0,
                                   0, 0.0446308, 0, 0, 0,
                                   0, 0, 0.0003793, 0, 0,
                                   0, 0, 0, 0.0381672, 0,
                                   0, 0, 0, 0, 0.0561985};
            double data_prec_3[] = {0.0047897, 0, 0, 0, 0,
                                   0, 0.0102444, 0, 0, 0,
                                   0, 0, 0.0011352, 0, 0,
                                   0, 0, 0, 0.1143574, 0,
                                   0, 0, 0, 0, 0.03827};

            cv::Mat static_prec_1 = cv::Mat(length,length,CV_64F,data_prec_1).clone();
            cv::Mat static_prec_2 = cv::Mat(length,length,CV_64F,data_prec_2).clone();
            cv::Mat static_prec_3 = cv::Mat(length,length,CV_64F,data_prec_3).clone();

            static_prec.insert(static_prec.end(),static_prec_1);
            static_prec.insert(static_prec.end(),static_prec_2);
            static_prec.insert(static_prec.end(),static_prec_3);

            break;
        }
        default: // TODO: add other colorspaces
        // to lahko tudi tako da ločiš c-jevske arraye in vse ostalo. arrayi lahko ostanejo v switchu, ostalo daš ven
            break;
    }
}