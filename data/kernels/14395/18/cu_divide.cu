#include "includes.h"
__global__ void cu_divide(float *numerator, float denominator, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
numerator[tid] = __fdividef(numerator[tid], denominator);
tid += stride;
}
}