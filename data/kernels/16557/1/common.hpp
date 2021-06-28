#ifndef __COMMON_HPP
#define __COMMON_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

typedef unsigned char uchar;
typedef unsigned int uint;

static void HandleError(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        std::stringstream ss;
        ss << line;
        std::string errMsg(cudaGetErrorString(err));
        errMsg += " (file: " + std::string(file);
        errMsg += " at line: " + ss.str() + ")";
        std::cout << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}


#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

#endif



