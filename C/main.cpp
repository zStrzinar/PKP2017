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
    // dokončano
    std::cout << "Detector initialized" << std::endl;

    std::cout << "Beginning frame-by-frame algorithm" << std::endl;
    int frame_number;
    cv::Mat frame_original, frame_resized, frame_colorspace;
    cv::Mat color_data_rows[3];
    cv::Size resized_size = cv::Size(em_image_size[0],em_image_size[1]);
    cv::Size original_size;

//    namedWindow("Original",WINDOW_AUTOSIZE);
//    namedWindow("Resized",WINDOW_AUTOSIZE);
//    namedWindow("YcrCb",WINDOW_AUTOSIZE);
    for(frame_number = 1; frame_number<=inputVideo.get(CV_CAP_PROP_FRAME_COUNT); frame_number++){
        // TODO: od tukaj naprej preverjaj ali se zadeva ujema z MATLABOM oziroma od kod naprej se ne!
        // začni kar s preverjenjem prvega frame-a
        inputVideo >> frame_original;
        resize(frame_original,frame_resized,resized_size);
        switch (colorSpace){
            case YCRCB:{
                cv::cvtColor(frame_resized, frame_colorspace, CV_BGR2YCrCb);
                break;
            }
            default:{
                std::cerr<<"Unsupported colorspace!"<<std::endl;
            }
        }

        cv::split(frame_colorspace,color_data_rows); // color_date_rows[0] je zdaj prvi kanal drame_colorspace
        color_data_rows[0] = color_data_rows[0].reshape(0,1).clone(); // color_data_rows[0] rata vrstična matrika. Prvih 50 vrednosti je iz prve vrstice originala, naslednih 50 je iz naslednje vrstice originala itd.
        color_data_rows[1] = color_data_rows[1].reshape(0,1).clone();
        color_data_rows[2] = color_data_rows[2].reshape(0,1).clone();

        cv::Mat color_data;//(3,em_image_size[0]*em_image_size[1],CV_64F);
        cv::vconcat(color_data_rows,3,color_data);
        color_data.convertTo(color_data,CV_64F);
        // Zlepim skupaj color_data od frame in spatial_data
        cv::Mat dataEM;//(5,em_image_size[0]*em_image_size[1],CV_64F);
        cv::vconcat(spatial_data,color_data,dataEM);
        original_size = frame_original.size();

        std::cout << color_data << std::endl;

        cv::Mat current_Mu[3], current_Cov[3], current_region;
        double current_w[3];

        if (frame_number==1){
            float df[] = {0,0.3,0.5,1};
            std::vector <float> vertical_ratio(df, df+sizeof(df)/sizeof(float) );
            std::transform(vertical_ratio.begin(), vertical_ratio.end(), vertical_ratio.begin(),
                           std::bind1st(std::multiplies<float>(),dataEM.cols));
//            vertical_ratio[3]+=1;
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

        }
        std::cout << "Frame " << frame_number << " done" << std::endl;
    }
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
