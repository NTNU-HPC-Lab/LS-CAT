#include "includes.h"
__global__ void inter_kernel(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < (NX+NY)*B){
int b = i / (NX+NY);
int j = i % (NX+NY);
if (j < NX){
OUT[i] = X[b*NX + j];
} else {
OUT[i] = Y[b*NY + j - NX];
}
}
}