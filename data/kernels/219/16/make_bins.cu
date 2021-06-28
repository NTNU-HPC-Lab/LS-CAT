#include "includes.h"
__global__ void make_bins(float *vec, int *bin, const int num_bins, const int n, const float slope, const float intercept)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

if ( xIndex < n ){
int bin_new_val;
float temp = abs(vec[xIndex]);
if ( temp > (intercept *.000001) ){
bin_new_val=slope * (intercept - temp);
}
else bin_new_val = num_bins;
bin[xIndex]=bin_new_val;
}
}