// Gualberto Casas
// A00942270

#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common.h"
#include <cuda_runtime.h>

using namespace std;

__global__ void equal(unsigned char* mat, unsigned char* out, int cols, int rows, int grayWidthStep, int *hist) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = iy * grayWidthStep + ix;
    if ((ix < cols) && (iy < rows)) out[tid] = hist[mat[tid]];
}

__global__ void generateHist(unsigned char* mat, int cols, int rows, int colorWidthStep, int *hist) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;	
    int ixy = threadIdx.x + threadIdx.y * blockDim.x;
    int tid = iy * colorWidthStep + ix;
    int size = 256;

    __shared__ int _hist[size];

    if (ixy < size) {
        _hist[ixy] = 0;
        /* __syncthreads(); */
    }

    if (ix < cols && iy < rows) {
        atomicAdd(&_hist[mat[tid]], 1);
        /* __syncthreads(); */
    }

    if (ixy < size) {
        atomicAdd(&hist[ixy], _hist[ixy]);
        /* __syncthreads(); */
    }

    __syncthreads();
}

__global__ void normal(int *hist) {
    int ixy = threadIdx.x + threadIdx.y * blockDim.x;
    int size = 256;

    __shared__ int _hist[size];

    if(ixy < size && blockIdx.x == 0 && blockIdx.y == 0) {
        _hist[ixy] = 0;
        _hist[ixy] = hist[ixy];
        __syncthreads();

        unsigned int normal = 0;
        for(int i = 0; i <= ixy; i++) normal += _hist[i];
        hist[ixy] = normal / 255;
    }
}

void equalWrapper(cv::Mat& mat, cv::Mat& out) {
    int *hist;
    int blockSize = 32;
    unsigned char *d_mat, *d_out;
    float graySize = mat.rows * mat.cols;
    size_t histB = sizeof(int) * 256;
    size_t grayB = out.step * out.rows;

    // Allocate device memory
    SAFE_CALL(cudaMalloc<unsigned char>(&mat, grayB), "CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<unsigned char>(&out, grayB), "CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc(&hist, histB), "CUDA Malloc failed");

    // Copy data from host to device
    SAFE_CALL(cudaMemcpy(d_mat, mat.ptr(), grayB, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(d_out, out.ptr(), grayB, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

    int gridX = ceil((float)input.cols / block.x);
    int gridY = ceil((float)input.rows/ block.y);
    const dim3 grid(gridX, gridY);
    const dim3 = (blockSize, blockSize);

    generateHist <<<grid, block>>>(d_mat, mat.cols, mat.rows, mat.step, hist);
    normal <<<grid, block>>>(h_s);
    equal <<<grid, block >>>(d_mat, d_out, mat.cols, mat.rows, mat.step, hist);

    SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

    // Copy data from device to host
    SAFE_CALL(cudaMemcpy(out.ptr(), d_out, grayB, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

    // Free the device memory
    SAFE_CALL(cudaFree(d_mat), "CUDA Free Failed");
    SAFE_CALL(cudaFree(d_out), "CUDA Free Failed");
    SAFE_CALL(cudaFree(hist), "CUDA Free FAiled");
}

// Same main as the CPU version
int main(int argc, char *argv[]) {
    // Load image
    string imagePath = "Images/dog2.jpeg";
    cv::Mat mat = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
    if (mat.empty()) cout << "Image Not Found!" << std::endl;

    // Declare out and grayscale image, and generate grayscale
    cv::Mat gray(mat.rows, mat.cols, CV_8UC1);
    cv::Mat out(mat.rows, mat.cols, CV_8UC1);
    cv::cvtColor(mat, gray, CV_BGR2GRAY);

    auto start_cpu =  chrono::high_resolution_clock::now();
    equalWrapper(gray, out);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    printf("Elapsed time: %f ms\nBlock size (%d, %d)\n", duration_ms.count(), xBlock, yBlock);

    // Display images
    /* namedWindow("Input", cv::WINDOW_NORMAL); */
    /* namedWindow("Output", cv::WINDOW_NORMAL); */
    /* imshow("Input", gray); */
    /* imshow("Output", out); */
    /* cv::waitKey(); */

    return 0;
}
