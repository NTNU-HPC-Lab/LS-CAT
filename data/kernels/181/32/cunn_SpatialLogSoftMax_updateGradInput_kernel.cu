#include "includes.h"
__global__ void cunn_SpatialLogSoftMax_updateGradInput_kernel(float *gradInput, float *output, float *gradOutput, int feature_size, int spatial_size, int data_size)
{
int idx = (threadIdx.x + blockDim.x*blockIdx.x);
idx = (idx/spatial_size)*feature_size + idx % spatial_size;

if (idx < data_size) {
int next_idx = idx + feature_size;
float gradSum = 0.0;
// Compute the sum of gradients
for(int i = idx; i < next_idx; i += spatial_size){
gradSum += gradOutput[i];
}
// Compute the new gradient
for(int i = idx; i < next_idx; i += spatial_size){
gradInput[i] = gradOutput[i] - __expf(output[i])*gradSum;
}
}
}