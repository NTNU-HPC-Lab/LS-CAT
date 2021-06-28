#include "includes.h"
__global__ void threshold_one(float *vec, float *vec_thres, int *bin, const int k_bin, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
// xIndex is a value from 1 to k from the vector ind

if ( (xIndex < n) & (bin[xIndex]<=k_bin) )
vec_thres[xIndex]=vec[xIndex];
}