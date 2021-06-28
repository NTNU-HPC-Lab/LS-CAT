#include "includes.h"

__constant__ float *c_Kernel;

__global__ void convolutionRowsKernel_v1( float *d_Dst, float *d_Src, int imageW, int filter_Rad, int Halo_steps )
{
extern __shared__ float s_Data[];

//Offset to the left halo edge
const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - Halo_steps) * ROWS_BLOCKDIM_X + threadIdx.x;
const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

d_Src += baseY * imageW + baseX;
d_Dst += baseY * imageW + baseX;

//Load main data
/*#pragma unroll

for (int i = Halo_steps; i < Halo_steps + ROWS_RESULT_STEPS; i++)
{
s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
}*/

//Load left halo
#pragma unroll
for (int i = 0; i < Halo_steps; i++)
{
s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
}

//Load right halo and main data
#pragma unroll
for (int i = Halo_steps; i < Halo_steps + ROWS_RESULT_STEPS + Halo_steps; i++)
{
s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
}

//Compute and store results
__syncthreads();
#pragma unroll
for (int i = Halo_steps; i < Halo_steps + ROWS_RESULT_STEPS; i++)
{
float sum = 0;
if (baseX + i * ROWS_BLOCKDIM_X < imageW)
{
#pragma unroll
for (int j = -filter_Rad; j <= filter_Rad; j++)
{
sum += c_Kernel[filter_Rad - j] * s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X + j];
}

d_Dst[i * ROWS_BLOCKDIM_X] = sum;

}
}
}