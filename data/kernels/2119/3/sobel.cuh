#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"
#include "device_launch_parameters.h"
#include "parameters.cuh"

#ifndef SOBEL_CUH
#define SOBEL_CUH

__global__ void gpu_sobel_kernel_naive(u_char *Source, u_char *Resultat, unsigned width, unsigned height);
__global__ void gpu_sobel_kernel_shared(u_char *Source, u_char *Resultat, unsigned width, unsigned height);
void cpu_sobel(u_char **Source, u_char **Resultat, unsigned width, unsigned height);

#endif