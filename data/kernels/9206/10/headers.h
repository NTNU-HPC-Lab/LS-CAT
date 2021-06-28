#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void process_kernel1(const float*, const float*, float*, int);

__global__ void process_kernel2(const float*, float*, int);

__global__ void process_kernel3(const float*, float*, int);