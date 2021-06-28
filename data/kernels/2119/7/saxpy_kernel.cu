#include "includes.h"
__global__ void saxpy_kernel(int n, float a, float *v1, float *v2, float *s){
int i = blockIdx.x*blockDim.x + threadIdx.x;
if ( i < n ) s[i] = a*v1[i] + v2[i];
}