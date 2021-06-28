#ifndef TRANSPO_CUH
#define TRANSPO_CUH

#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"
#include "device_launch_parameters.h"
#include "parameters.cuh"


__global__ void gpu_transpo_kernel_naive(u_char *Source, u_char *Resultat, unsigned width, unsigned height);
__global__ void gpu_transpo_kernel_shared(u_char *Source, u_char *Resultat, unsigned height, unsigned width);
void cpu_transpo(u_char **Source, u_char **Resultat, unsigned width, unsigned height);



#endif