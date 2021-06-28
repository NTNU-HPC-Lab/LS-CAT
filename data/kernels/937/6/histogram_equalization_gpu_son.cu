#include "includes.h"
__global__ void histogram_equalization_gpu_son (unsigned char * d_in, unsigned char * d_out, int * d_lut, int img_size,  int serialNum)
{
int x = threadIdx.x + blockDim.x*blockIdx.x;
if (x >= img_size) return;

d_out[x] = (unsigned char) d_lut[d_in[x]];
}