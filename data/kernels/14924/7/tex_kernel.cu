#include "includes.h"
__global__ void tex_kernel(cudaTextureObject_t texture_obj, int num_samples, float* output) {
unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
if (idx < num_samples) {
float u = idx / static_cast<float>(num_samples);
output[idx] = tex1D<float>(texture_obj, u);
}
}