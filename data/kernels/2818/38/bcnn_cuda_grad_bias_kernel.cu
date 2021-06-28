#include "includes.h"
__global__ void bcnn_cuda_grad_bias_kernel(float *grad_bias, float *grad_data, int num_channels, int spatial_size) {
int offset = blockIdx.x * blockDim.x + threadIdx.x;
int channel = blockIdx.y;
int batch_size = blockIdx.z;

if (offset < spatial_size)
grad_bias[channel] +=
grad_data[(batch_size * num_channels + channel) * spatial_size +
offset];
}