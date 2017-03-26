cmake_minimum_required(VERSION 3.6)
project(PKP2017_SEGMENTATION)

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(C ${SOURCE_FILES})

target_link_libraries( C ${OpenCV_LIBS} )