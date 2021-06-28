#include "includes.h"
__global__ void rotatewin(float* aframe2, float *aframe, float *win, int N, int offset){
int k = threadIdx.x + blockIdx.x*blockDim.x;
aframe2[(k+offset)%N] = win[k]*aframe[k];
}