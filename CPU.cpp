// Gualberto Casas
// A00942270
// g++ -o CPU CPU.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -std=c++11

#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

// ******************************************************
// *********************** CPU **************************
// ******************************************************

void equal(long *hist, cv::Mat& mat, cv::Mat& out) {
    for (int i = 0; i < mat.rows; i += 1) {
        for (int j = 0; j < mat.cols; j += 1) {
            out.at<uchar>(i, j) = hist[(int)mat.at<uchar>(i, j)];
        }
    }
}

void normal(long *hist, cv::Mat& mat) {
    int size = 256;
    // Reset Histogram
    // Create temporary array
    long arr[size] = {};
    for (int i = 0; i < size; i += 1) {
        arr[i] = hist[i];
        hist[i] = 0;
    }

    // Normalize
    for (int i = 0; i < size; i += 1) {
        for (int j = 0; j <= i; j += 1) hist[i] += arr[j];
        hist[i] = (hist[i] * 255) / (mat.rows * mat.cols);
    }
}

void generateHist(long *hist, cv::Mat& mat) {
    for (int i = 0; i < mat.rows; i += 1) {
        for (int j = 0; j < mat.cols; j += 1) {
            hist[(int)mat.at<uchar>(i, j)] += 1;
        }
    }
}

void equalWrapper(cv::Mat& mat, cv::Mat& out) {
    long hist[256] = {};
    generateHist(hist, mat);
    normal(hist, mat);
    equal(hist, mat, out);
}

int main(int argc, char *argv[]) {
    // Load image
    string img = "Images/dog1.jpeg";
    cv::Mat mat = cv::imread(img, CV_LOAD_IMAGE_COLOR);
    if (mat.empty()) cout << "Image Not Found!" << std::endl;

    // Declare output and grayscale image, and generate grayscale
    cv::Mat gray(mat.rows, mat.cols, CV_8UC1);
    cv::Mat out(mat.rows, mat.cols, CV_8UC1);
    cv::cvtColor(mat, gray, CV_BGR2GRAY);

    auto start_cpu =  chrono::high_resolution_clock::now();
    equalWrapper(gray, out);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    printf("Elapsed time: %f ms\n", duration_ms.count());

    // Display images
    /* namedWindow("Input", cv::WINDOW_NORMAL); */
    /* namedWindow("Output", cv::WINDOW_NORMAL); */
    /* imshow("Input", mat); */
    /* imshow("Output", out); */
    /* cv::waitKey(); */

    return 0;
}
