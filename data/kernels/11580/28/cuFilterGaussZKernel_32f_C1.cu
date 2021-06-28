#include "includes.h"
__global__ void cuFilterGaussZKernel_32f_C1(float* dst, float* src, const int y, const int width, const int depth, const size_t stride, const size_t slice_stride, float sigma, int kernel_size)
{
int x = blockIdx.x*blockDim.x + threadIdx.x;
int z = blockIdx.y*blockDim.y + threadIdx.y;

if(x>=0 && z>= 0 && x<width && z<depth)
{
float sum = 0.0f;
int half_kernel_elements = (kernel_size - 1) / 2;

// convolve horizontally
float g0 = 1.0f / (sqrtf(2.0f * 3.141592653589793f) * sigma);
float g1 = exp(-0.5f / (sigma * sigma));
float g2 = g1 * g1;
sum = g0 * src[z*slice_stride + y*stride + x];
float sum_coeff = g0;
for (int i = 1; i <= half_kernel_elements; i++)
{
g0 *= g1;
g1 *= g2;
int cur_z = fmaxf(0, fminf(depth-1, z + i));
sum += g0 * src[cur_z*slice_stride + y*stride + x];
cur_z = fmaxf(0, fminf(depth-1, z - i));
sum += g0 * src[cur_z*slice_stride + y*stride + x];
sum_coeff += 2.0f*g0;
}
dst[z*slice_stride + y*stride + x] = sum/sum_coeff;
}
}