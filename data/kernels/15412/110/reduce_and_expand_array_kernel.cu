#include "includes.h"
__global__  void reduce_and_expand_array_kernel(const float *src_gpu, float *dst_gpu, int current_size, int groups)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;

if (index < current_size) {
float val = 0;
for (int i = 0; i < groups; ++i) {
val += src_gpu[index + i*current_size];
}
for (int i = 0; i < groups; ++i) {
dst_gpu[index + i*current_size] = val / groups;
}
}
}