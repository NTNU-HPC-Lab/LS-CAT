#include "includes.h"
__device__ void softmax_device_new_api(float *input, int n, float temp, int stride, float *output)
{
int i;
float sum = 0;
float largest = -INFINITY;
for (i = 0; i < n; ++i) {
int val = input[i*stride];
largest = (val>largest) ? val : largest;
}
for (i = 0; i < n; ++i) {
float e = expf(input[i*stride] / temp - largest / temp);
sum += e;
output[i*stride] = e;
}
for (i = 0; i < n; ++i) {
output[i*stride] /= sum;
}
}
__device__ void softmax_device(int n, float *input, float temp, float *output)
{
int i;
float sum = 0;
float largest = -INFINITY;
for(i = 0; i < n; ++i){
int val = input[i];
largest = (val>largest) ? val : largest;
}
for(i = 0; i < n; ++i){
float e = exp(input[i]/temp - largest/temp);
sum += e;
output[i] = e;
}
for(i = 0; i < n; ++i){
output[i] /= sum;
}
}
__global__ void softmax_tree_kernel(float *input, int spatial, int batch, int stride, float temp, float *output, int groups, int *group_size, int *group_offset)
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (id >= spatial*batch*groups) return;
int s = id % spatial;
id = id / spatial;
int g = id % groups;
int b = id / groups;
int goff = group_offset[g] * spatial;
int boff = b*stride;
softmax_device_new_api(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);
}