#include "includes.h"

__global__ void add_matrices(float *ad,float *bd,float *cd,int N)
{
cd[threadIdx.y * N + threadIdx.x] = ad[threadIdx.y * N + threadIdx.x] + bd[threadIdx.y * N + threadIdx.x];
}