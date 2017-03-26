#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Hello, World!" << std::endl;

    if(argc != 2){
        std::cout << "Wrong number of parameters!" << std::endl;
        return -1;
    }

    std::cout << "Loading input image: " << argv[1] << std::endl;
    cv::Mat input;
    input = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

    std::cout << "Detecting edges in input image" << std::endl;
    cv::Mat edges;
    cv::Canny(input,edges, 10, 100);

    cv::namedWindow("Original",CV_WINDOW_AUTOSIZE);
    cv::imshow("Original", input);

    cv::waitKey(0);
    return 0;
}