#include "includes.h"
__global__ void LinearBinning(float *vec, int *bin, int *bin_counters, const int num_bins, const int MaxBin, const int n, const float slope, const float intercept)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
float temp = abs(vec[xIndex]);
if ( xIndex < n ){
if ( temp > (intercept *.000001) ){
bin[xIndex]=max(0.0f,slope * (intercept - temp));
if (bin[xIndex]<MaxBin) atomicAdd(bin_counters+bin[xIndex],1);
}
else bin[xIndex] = slope * intercept + 1.0f;
}
}