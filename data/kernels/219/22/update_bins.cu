#include "includes.h"
__global__ void update_bins(float *vec, int *bin, int *bin_counters, const int num_bins, const int n, const float slope, const float intercept)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

if ( xIndex < n ){
int bin_new_val;
float temp = abs(vec[xIndex]);
if ( temp > (intercept *.000001) ){
bin_new_val=slope * (intercept - temp);
}
else bin_new_val = num_bins;

if ( bin[xIndex] != bin_new_val ){
if (bin[xIndex] < num_bins)
atomicAdd(bin_counters+bin[xIndex],-1);
if ( bin_new_val < num_bins )
atomicAdd(bin_counters+bin[xIndex],1);
bin[xIndex]=bin_new_val;
}


}
}