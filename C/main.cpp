//
// Created by ziga on 8.4.2017.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void printHelp() {
    std::string msg;
    msg = "Help\n\tUsage:\n\t\t-h\t .......... for help\n\t\tpath\t ........... specify video path\n\t\tpath output\t ... specify video path and output path";
    std::cout << msg << std::endl;
}

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


}