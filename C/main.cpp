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
#include "test.h"
#include "showResults.h"
#include "zStrzinar.h"

using namespace cv;
using namespace std;

void printHelp();
void loadPriorModelFromDisk(Colorspace colorSpace, std::vector<cv::Mat> &mix_Mu, std::vector<cv::Mat> &mix_Cov, std::vector<cv::Mat> &mix_w, std::vector<cv::Mat> &static_prec);
void getSpacialData(cv::Size em_image_size, cv::Mat& spatial_data);
void momentMatchPdf(cv::Mat previous_Mu, cv::Mat current_Mu, cv::Mat previous_Cov, cv::Mat current_Cov, std::vector<float> current_w, cv::Mat& new_Mu, cv::Mat& new_Cov, cv::Mat& new_w);

int main (int argc, char ** argv){
    // testiranje();
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

    // Settings
    long double eps = 2.2204e-16;
    bool use_prior_on_mixture = true; // MATLAB: example.m:39 % detector constructor
    bool use_uniform_component = true; // MATLAB: example.m:39 % detector constructor
    Colorspace colorSpace = YCRCB; // MATLAB: example.m:39 % detector constructor
    int maxEMsteps = 10; // MATLAB: example.m:39 % detector constructor
    cv::Mat PI_i = cv::Mat(); // MATLAB: example.m:39 % detector constructor
    std::vector<cv::Mat> current_mix_Mu, current_mix_Cov, current_mix_W, current_mix_Prec;
    std::vector<cv::Mat> prior_mix_Mu, prior_mix_Cov, prior_mix_W, prior_mix_Prec;
    loadPriorModelFromDisk(colorSpace, prior_mix_Mu, prior_mix_Cov, prior_mix_W, prior_mix_Prec); // hardcoded values
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
    cv::namedWindow("Moje okno",WINDOW_AUTOSIZE);
    for(frame_number = 1; frame_number<=inputVideo.get(CV_CAP_PROP_FRAME_COUNT); frame_number++){
        inputVideo.set(CV_CAP_PROP_POS_FRAMES, frame_number-1);
        inputVideo >> frame_original;

        char currentFile[17];
        sprintf(currentFile, "images/%05d.jpg",frame_number);

        std::cout << currentFile << std::endl;
        frame_original = imread(currentFile, CV_LOAD_IMAGE_COLOR); // TODO: to je samo za debugiranje!
        original_size = frame_original.size();

        cv::resize(frame_original,frame_resized,resized_size,0.5,0.5,INTER_LINEAR);

        switch (colorSpace){
            case YCRCB:{
                cv::Mat Y,Cr,Cb; std::vector<cv::Mat> bgr,YCrCb;
                cv::split(frame_resized, bgr);
                Y = 0.0593*bgr[2]+(1-0.0593-0.2627)*bgr[1]+0.2627*bgr[1];
                cv::cvtColor(frame_resized, frame_colorspace, CV_BGR2YCrCb);
                std::cout << std::endl << std::endl << std::endl << Y << std::endl << std::endl << std::endl << std::endl;
                //std::cout << std::endl << std::endl << std::endl << frame_colorspace << std::endl << std::endl << std::endl << std::endl;
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
        cv::Mat current_Mu[3], current_Cov[3], current_region;
        if (frame_number==1){
            float df[] = {0,0.3,0.5,1}; // to so rocno nastavljene zacetne meje med območji
            std::vector <float> vertical_ratio(df, df+sizeof(df)/sizeof(float) ); // to samo ustvari vektor in ga zafila z vrednostmi df[]
            std::transform(vertical_ratio.begin(), vertical_ratio.end(), vertical_ratio.begin(),
                           std::bind1st(std::multiplies<float>(),dataEM.cols)); // to pomnoži vektor z dataEM.cols
            // oboje skupaj je v matlabu: vertical_ratio = df*size(dataEM,2)

            int k;
            for (k=0; k<=2; k++){ // cez vse regije:
                current_region = dataEM.colRange((int)vertical_ratio[k],(int)vertical_ratio[k+1]); // v dataEM so vse lokaije kar po vrsti v vrstici
                // TODO: ali je tukaj (eno vrstico gor) pravilen range? vključenost prvega in zadnjega stolpca v primerjavi z Matlabom?

                cv::calcCovarMatrix(current_region, current_Cov[k], current_Mu[k], CV_COVAR_NORMAL|CV_COVAR_COLS); // Kovariančna matrika in srednja vrednost
                current_Cov[k] = current_Cov[k] / (current_region.cols - 1);
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
                    cv::calcCovarMatrix(current_region, current_Cov[k], current_Mu[k], CV_COVAR_NORMAL|CV_COVAR_COLS, current_region.type()); // Kovariančna matrika in srednja vrednost
                    current_Cov[k] = current_Cov[k] / (current_region.cols - 1);

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

        std::cout << "Frame " << frame_number << " done" << std::endl;
    }
    return 0;
}

void printHelp() {
    std::string msg;
    msg = "Help\n\tUsage:\n\t\t-h\t .......... for help\n\t\tpath\t ........... specify video path\n\t\tpath output\t ... specify video path and output path";
    std::cout << msg << std::endl;
}
void loadPriorModelFromDisk(Colorspace colorSpace, std::vector<cv::Mat> &mix_Mu, std::vector<cv::Mat> &mix_Cov,
                            std::vector<cv::Mat> &mix_w, std::vector<cv::Mat> &static_prec) {
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
void getSpacialData(cv::Size em_image_size, cv::Mat& spatial_data){
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
    cv::vconcat(prvaVrstica,drugaVrstica,spatial_data);
    spatial_data = spatial_data.clone();
}
void momentMatchPdf(cv::Mat previous_Mu, cv::Mat current_Mu, cv::Mat previous_Cov, cv::Mat current_Cov, std::vector<float> current_w, cv::Mat& new_Mu, cv::Mat& new_Cov, cv::Mat& new_w){
    float sum_w=0;
    int i;
    for (i=0; i<current_w.size(); i++){
        sum_w+=current_w[i];
    }
    for (i=0; i<current_w.size(); i++){
        current_w[i]/=sum_w;
    }

    // previous_mu in current_mu sta dve stolpični matriki. Moramo ju zlepit skupaj.
    cv::Mat Mu; // za zlepljena _mu
    cv::hconcat(previous_Mu,current_Mu,Mu);

    cv::Mat Multi;
    Multi = Mat::zeros(Mu.rows, Mu.cols, Mu.type());

    for (i=0; i<Mu.cols; i++){
        Multi.col(i) = Mu.col(i)*current_w[i];
    }

    cv::Mat new_mu;
    cv::reduce(Multi, new_mu, 1, CV_REDUCE_SUM); // sešteje vsako vrstico posebej da dobi elemente novege stolpične matrike. argument 1 pomeni da bo rezultat STOLPIČNA matrika

    new_Mu = current_w[0]*current_Mu + current_w[1]*previous_Mu;

    cv::Mat temporary1, temporary2;
    temporary1 = previous_Cov+previous_Mu*previous_Mu.t();
    temporary2 = current_Cov+current_Mu*current_Mu.t();
    new_Cov = current_w[0]*temporary1 + current_w[1]*temporary2-new_mu*new_mu.t();

    new_w = cv::Mat(1,1,CV_64F,sum_w);
}