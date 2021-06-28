#include "includes.h"

__constant__ float *c_Kernel;

__global__ void convolutionColumnsKernel_up_smp( float *d_Dst, float *d_Src, int imageW, int imageH, int n_imageH, int pitch, int filter_Rad, int Halo_steps )
{
extern __shared__ float s_Data[];

//Offset to the upper halo edge
const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - Halo_steps) * COLUMNS_BLOCKDIM_Y + threadIdx.y;

if (baseX < imageW)
{
d_Src += baseY * pitch + baseX;
d_Dst += 2 * baseY * pitch + baseX;

//Upper halo
//#pragma unroll
for (int i = 0; i < Halo_steps; i++)
{
s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
}

//Lower halo + Main data
//#pragma unroll
for (int i = Halo_steps; i < Halo_steps + COLUMNS_RESULT_STEPS + Halo_steps; i++)
{
s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
}

//Compute and store results
__syncthreads();
//#pragma unroll
for (int i = Halo_steps; i < COLUMNS_RESULT_STEPS + Halo_steps; ++i)
{
int Pos_y = 2 * baseY + (2 * i) * COLUMNS_BLOCKDIM_Y;

if (Pos_y < n_imageH)
{
float sum_1 = 0.0f, sum_2 = 0.0f;

//#pragma unroll
for (int l = -(filter_Rad / 2); l <= filter_Rad / 2; ++l)
{
int t = 2 * l;

float temp = s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y + l];

sum_1 += c_Kernel[filter_Rad + t] * temp * 2.0f;
sum_2 += c_Kernel[filter_Rad + t - 1] * temp * 2.0f;
}

sum_2 += c_Kernel[2 * filter_Rad] * 2.0f * s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y + filter_Rad / 2 + 1];

d_Dst[2 * i * COLUMNS_BLOCKDIM_Y * pitch] = sum_1;
if (Pos_y + 1 < n_imageH)d_Dst[2 * i * COLUMNS_BLOCKDIM_Y * pitch + pitch] = sum_2;

}

}
}
}