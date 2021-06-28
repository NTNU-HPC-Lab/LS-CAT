#include "includes.h"
__global__ void mean_kernel(int n, float* v1, float* v2, float* res){
int i = threadIdx.x + blockIdx.x*blockDim.x;
if( i < n ) res[i] = (v1[i] + v2[i])/2;
}