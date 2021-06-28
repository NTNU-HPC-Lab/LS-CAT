#include "includes.h"

__constant__ float *c_Kernel;

__global__ void convolutionColumnsKernel_v1( float *d_Dst, float *d_Src, int imageW, int imageH, int pitch, int filter_Rad, int Halo_steps )
{
extern __shared__ float s_Data[];

//Offset to the upper halo edge
const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - Halo_steps) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
d_Src += baseY * pitch + baseX;
d_Dst += baseY * pitch + baseX;

/*	//Main data
#pragma unroll

for (int i = Halo_steps; i < Halo_steps + COLUMNS_RESULT_STEPS; i++)
{
s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS+2*Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
}*/

//Upper halo
#pragma unroll
for (int i = 0; i < Halo_steps; i++)
{
s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
}

//Lower halo + Main data
#pragma unroll
for (int i = Halo_steps; i < Halo_steps + COLUMNS_RESULT_STEPS + Halo_steps; i++)
{
s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
}

//Compute and store results
__syncthreads();
#pragma unroll
for (int i = Halo_steps; i < Halo_steps + COLUMNS_RESULT_STEPS; i++)
{
float sum = 0;

if (baseY + i * COLUMNS_BLOCKDIM_Y < imageH)
{
#pragma unroll
for (int j = -filter_Rad; j <= filter_Rad; j++)
{
sum += c_Kernel[filter_Rad - j] * s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
}

d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
}
}
}