#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#if defined(__GNUC__)
#define SafeCall(expr) ___SafeCall(expr, __FILE__, __LINE__, __func__)
#else
#define SafeCall(expr) ___SafeCall(expr, __FILE__, __LINE__)
#endif



static inline void error(const char *error_string, const char *file, const int line, const char *func)
{
    std::cout << "Error: " << error_string << "\t" << file << ":" << line << std::endl;
    exit(0);
}

static inline void ___SafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
{
    if (cudaSuccess != err)
        error(cudaGetErrorString(err), file, line, func);
}

template <class T>
__global__ void callDeviceFunctor(const T functor)
{
    functor();
}


#endif