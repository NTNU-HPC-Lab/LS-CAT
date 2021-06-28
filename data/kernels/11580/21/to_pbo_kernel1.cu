#include "includes.h"
__global__ void to_pbo_kernel1(unsigned char* g_in, int stride_in, uchar4* g_out, int stride_out, int width, int height)
{
const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;

if (x<width && y<height)
{
unsigned char value = g_in[y*stride_in+x];
g_out[y*stride_out+x] = make_uchar4(value, value, value, 1);
}
}