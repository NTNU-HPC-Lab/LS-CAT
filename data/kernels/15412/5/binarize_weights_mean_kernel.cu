#include "includes.h"
__global__ void binarize_weights_mean_kernel(float *weights, int n, int size, float *binary, float *mean_arr_gpu)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int f = i / size;
if (f >= n) return;
float mean = mean_arr_gpu[f];
binary[i] = (weights[i] > 0) ? mean : -mean;
}