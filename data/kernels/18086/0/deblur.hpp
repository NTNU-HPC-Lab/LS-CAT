#ifndef DEBLUR_HPP
#define DEBLUR_HPP
#include <iostream>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#define threadsPerBlock 16
#define gpuErrChk(ans) {gpuAssert((ans), __FILE__, __LINE__);}
#define getMoment std::chrono::high_resolution_clock::now()
#define getTimeElapsed(end, start) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0

#define getExeTime(mess, ans) {auto start = getMoment;ans;auto end = getMoment; std::cout <<mess<< getTimeElapsed(end, start)<<std::endl;}
inline void gpuAssert(cudaError_t code, const char *file, int64_t line, bool abort = true)
{
    if(code!=cudaSuccess)
    {
        fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

inline void printData(float *matrix, int row, int col, char *name)
{
    for(int y = 0; y < row; y++)
    {
        for(int x = 0; x < col; x++)
        {
            printf("%s[%d] = %f\n", name, y * col + row, matrix[y * col + row]);
        }
    }
}
void cuda_RichardsonLucy(float *blurredImg, float *psf, float *result, int blurredImg_rows, int blurredImg_cols,
                         int psf_rows, int psf_cols, int iter);
//template<typename T> void printMatrix(T * matrix, int row, int col);
//cv::Mat cuda_RichardsonLucy(cv::Mat blurredImg, cv::Mat psf, int iterations);

#endif // DEBLUR_HPP
