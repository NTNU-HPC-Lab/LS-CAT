#ifndef DOTP_CUH
#define DOTP_CUH

#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"
#include "device_launch_parameters.h"

__global__ void gpu_dotp_kernel(int n, float* vec1, float* vec2, float* res);
float cpu_dotp(float* vec1, float* vec2, int size);

#endif