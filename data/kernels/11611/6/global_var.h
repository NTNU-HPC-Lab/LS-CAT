#ifndef GLOBAL_HH
#define GLOBAL_HH

#pragma once

#define LEVELS 3

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

#include <stdio.h>
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        printf( "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


#define HALF_WIN                                4
#define PATCH_SIZE_L                            (HALF_WIN*2+1)

#define HALF_PATCH_SIZE_WITH_BORDER             5
#define PATCH_SIZE_WITH_BORDER                  (HALF_PATCH_SIZE_WITH_BORDER*2+1)

#define PATCH_MAX_CENTER                        8
#define PATCH_SIZE_MAX                          17

#define NB_FEATURE_MAX                          120
#define THRESHOLD                               0.01


#endif // GLOBAL_HH

