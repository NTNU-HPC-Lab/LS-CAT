#include "includes.h"
__global__ void sum3(double *d_result, double *d_a, double *d_b, double *d_c, int dSize){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < dSize) {
d_result[tid] = d_a[tid] + d_b[tid] +d_c[tid];
tid  += blockDim.x * gridDim.x;
}
}