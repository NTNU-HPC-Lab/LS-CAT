#include "includes.h"

/*Performs separable convolution on 3d cube*/



__global__ void convolution_sep(float *output, const float *input, const float *kernel, const int kernel_size, const dim3 imsize, int dir)
{
size_t ix, iy, iz;
if (dir == X_DIR)
{
ix = blockDim.x*blockIdx.x + threadIdx.x;
iy = blockDim.y*blockIdx.y + threadIdx.y;
iz = blockIdx.z;
}
else if (dir == Y_DIR)
{
iy = blockDim.x*blockIdx.x + threadIdx.x;
ix = blockDim.y*blockIdx.y + threadIdx.y;
iz = blockIdx.z;
}
else if (dir == EPS_DIR)
{
iz = blockDim.x*blockIdx.x + threadIdx.x;
ix = blockDim.y*blockIdx.y + threadIdx.y;
iy = blockIdx.z;
}

const bool valid = ix < imsize.x && iy < imsize.y && iz < imsize.z;
const size_t cube_idx = ix + iy*imsize.x + iz*imsize.x*imsize.y;

const size_t radius_size = kernel_size / 2;

extern __shared__ float s_image[]; //size is on kernel call
const size_t s_dim_x = blockDim.x + 2 * radius_size;
const size_t s_ix = radius_size + threadIdx.x;
const size_t s_iy = threadIdx.y;
float result = 0.0;

if (threadIdx.x < radius_size) //is on the left part of the shared memory
{
s_image[s_ix - radius_size + s_iy*s_dim_x] = 0.0f;
}
if (threadIdx.x >= (blockDim.x - radius_size)) //is on the right part
{
s_image[s_ix + radius_size + s_iy*s_dim_x] = 0.0f;
}



s_image[s_ix + s_iy*s_dim_x] = (valid) ? input[cube_idx] : 0.0f;


__syncthreads();


#pragma unroll
for (int i = 0; i < kernel_size; i++)
{
result += kernel[i] * s_image[s_ix - i + radius_size + s_iy*s_dim_x];
}

if (valid)
{

output[cube_idx] = result;
}
}