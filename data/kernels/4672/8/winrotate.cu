#include "includes.h"
__global__ void winrotate(float* inframe2, float* inframe, float *win, int N, int offset){
int k = (threadIdx.x + blockIdx.x*blockDim.x);
inframe2[k] = win[k]*inframe[(k+offset)%N];
}