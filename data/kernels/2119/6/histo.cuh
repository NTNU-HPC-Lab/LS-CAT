#ifndef HISTO_CUH
#define HISTO_CUH

#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"
#include "device_launch_parameters.h"
#include "parameters.cuh"

__global__ void gpu_histo_kernel_naive(u_char* Source, int *res, unsigned height, unsigned width);
__global__ void gpu_histo_kernel_shared(u_char* Source, int *res, unsigned height, unsigned width);
void cpu_histo(u_char** Source, int (*res)[256], unsigned height, unsigned width);

#endif