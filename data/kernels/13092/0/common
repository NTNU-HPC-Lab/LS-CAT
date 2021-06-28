//
// Created by deano on 28/04/16.
//

#pragma once
#ifndef VIZDOOM_COMMON_H
#define VIZDOOM_COMMON_H

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <memory>
#include <sstream>
#include <cstdint>
#include <iostream>

//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN

#define FatalError( s ) do {                                           \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
} while(0)

#define checkCUDNN( status ) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors( status ) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#if defined(USE_HALF_FLOATS)
#define CUDNN_DATA_HALF_OR_FLOAT CUDNN_DATA_HALF
typedef half half_or_float;
#else
#define CUDNN_DATA_HALF_OR_FLOAT CUDNN_DATA_FLOAT
typedef float half_or_float;
#endif

// sized index type
typedef uint_fast8_t TinyIndex;
typedef uint_fast16_t SmallIndex;
typedef uint_fast32_t LargeIndex;
typedef uint_fast64_t HugeIndex;

// default don't care Index to small type
typedef SmallIndex Index;

#endif //VIZDOOM_COMMON_H
