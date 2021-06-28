#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void ReduceRowMajor(int *g_idata, int *g_odata, int size);

__global__ void ReduceRowMajor2(int *g_idata, int *g_odata, int size);

__global__ void ReduceRowMajor3(int *g_idata, int *g_odata, int size);

__global__ void ReduceRowMajor5(int *g_idata, int *g_odata, int size);

__device__ void warpReduce(volatile int* sdata, int tid, int n);

