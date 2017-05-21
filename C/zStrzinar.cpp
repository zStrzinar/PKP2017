//
// Created by ziga on 8.5.2017.
//

#include "zStrzinar.h"

void printMat(std::string text,  cv::Mat matrika){
    std::cout << text << matrika << std::endl;
}

void removeRow(const cv::Mat &in, cv::Mat &out, int index){
    cv::Mat up, down;
    if(index>=0 && index < in.rows) {
        up = in.rowRange(0, index);
        down = in.rowRange(index + 1, in.rows);
        cv::hconcat(up, down, out);
    }
    else{
        std::cerr << "Index out of bounds! Index is " << index << " in.rows is " << in.rows << std::endl;
    }
    return;
}

void removeCol(const cv::Mat &in, cv::Mat &out, int index){
    cv::Mat left, right;
    if(index>=0 && index < in.cols) {
        left = in.colRange(0, index);
        right = in.colRange(index + 1, in.rows);
        cv::vconcat(left, right, out);
    }
    else{
        std::cerr << "Index out of bounds! Index is " << index << " in.cols is " << in.cols << std::endl;
    }
    return;
}

void removeRows(const cv::Mat &in, cv::Mat &out, std::vector<bool> deleteVector){
    int removed = 0;
    int i;
    cv::Mat current = in.clone();
    for(i=0, removed=0; i<deleteVector.size(); i++){
        if(deleteVector[i]){
            removeRow(current,current,i-removed);
            removed++;
        }
    }
    out = current;
}

void removeRows(const cv::Mat &in, cv::Mat &out, std::vector<int> deleteVector){
    std::vector<bool> boolDeleteVector;
    int i, max=-1;
    for (i=0; i<deleteVector.size(); i++){
        if (deleteVector[i]>max)
            max = deleteVector[i];
    }
    for (i=0; i<max; i++){
        boolDeleteVector.push_back(false);
    }
    for(i=0; i<deleteVector.size(); i++){
        boolDeleteVector[deleteVector[i]] = true;
    }
    removeRows(in,out,boolDeleteVector);
}

void removeCols(const cv::Mat &in, cv::Mat &out, std::vector<bool> deleteVector){
    int removed = 0;
    int i;
    cv::Mat current = in.clone();
    for(i=0, removed=0; i<deleteVector.size(); i++){
        if(deleteVector[i]){
            removeCol(current,current,i-removed);
            removed++;
        }
    }
    out = current;
}

void removeCols(const cv::Mat &in, cv::Mat &out, std::vector<int> deleteVector){
    std::vector<bool> boolDeleteVector;
    int i, max=-1;
    for (i=0; i<deleteVector.size(); i++){
        if (deleteVector[i]>max)
            max = deleteVector[i];
    }
    for (i=0; i<max; i++){
        boolDeleteVector.push_back(false);
    }
    for(i=0; i<deleteVector.size(); i++){
        boolDeleteVector[deleteVector[i]] = true;
    }
    removeCols(in,out,boolDeleteVector);
}

template <typename T> // TODO: to nekako ne dela
void removeVectorElements(const std::vector<T> &in, std::vector<T> &out, std::vector<bool> deleteVector){
    std::vector <T> output;
    int i;
    if (deleteVector.size()!=in.size()){
        std::cerr << "deleteVector and in vector are not the same length! deleteVector.size() = "  << deleteVector.size()
                  << " in.size() = " << in.size() << std::endl;
    }
    for (i=0; i<in.size(); i++){
        if(!deleteVector[i]){
            output.push_back(in[i]);
        }
    }
}

void removeVectorElementsInt(const std::vector<int> &in, std::vector<int> &out, std::vector<bool> deleteVector){
    std::vector <int> output;
    int i;
    if (deleteVector.size()!=in.size()){
        std::cerr << "deleteVector and in vector are not the same length! deleteVector.size() = "  << deleteVector.size()
                  << " in.size() = " << in.size() << std::endl;
    }
    for (i=0; i<in.size(); i++){
        if(!deleteVector[i]){
            output.push_back(in[i]);
        }
    }
}