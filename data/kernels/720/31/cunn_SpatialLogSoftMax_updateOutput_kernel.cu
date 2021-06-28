#include "includes.h"
__global__ void cunn_SpatialLogSoftMax_updateOutput_kernel (float *output, float *input, int feature_size, int spatial_size, int data_size, float constant)
{
int idx = (threadIdx.x + blockDim.x*blockIdx.x);
idx = (idx/spatial_size)*feature_size + idx % spatial_size;

if (idx < data_size) {
int next_idx = idx + feature_size;
float logsum = 0.0;
float max = -2e38;
// max
for(int i = idx; i < next_idx; i += spatial_size) {
if (max < input[i]) max = input[i];
}

// logsum
for(int i = idx; i < next_idx; i += spatial_size) {
if (!isnan(input[i])) {
logsum += __expf(input[i]-max);
}
}
logsum += constant;
logsum = __logf(logsum) + max;

// logsoftmax
for(int i = idx; i < next_idx; i += spatial_size){
output[i] = input[i] - logsum;
}
}
}