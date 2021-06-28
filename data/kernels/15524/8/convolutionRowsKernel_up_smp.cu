#include "includes.h"

__constant__ float *c_Kernel;

__global__ void convolutionRowsKernel_up_smp( float *d_Dst, float *d_Src, int imageW, int n_imageW, int imageH, int filter_Rad, int Halo_steps )
{
extern __shared__ float s_Data[];

//Offset to the left halo edge
const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - Halo_steps) * ROWS_BLOCKDIM_X + threadIdx.x;
const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
const int baseX1 = blockIdx.x * ROWS_RESULT_STEPS * 2 * ROWS_BLOCKDIM_X + 2 * threadIdx.x;

if (baseY < imageH)
{
d_Src += baseY * imageW + baseX;
d_Dst += baseY * n_imageW + baseX1;

//Load left halo
//#pragma unroll
for (int i = 0; i < Halo_steps; ++i)
{
s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
}

//Load right halo and main data
//#pragma unroll
for (int i = Halo_steps; i < Halo_steps + ROWS_RESULT_STEPS + Halo_steps; ++i)
{
s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
}

//Compute and store results
__syncthreads();

//#pragma unroll
for (int i = Halo_steps; i < Halo_steps + ROWS_RESULT_STEPS; ++i)
{
int pos_x = (baseX1 + 2 * (i - Halo_steps) * ROWS_BLOCKDIM_X);

if (pos_x < n_imageW)
{
float sum_1 = 0.0f, sum_2 = 0.0f;

//#pragma unroll
for (int l = -(filter_Rad / 2); l <= filter_Rad / 2; ++l)
{
int t = 2 * l;

float temp = s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X + l];
sum_1 += c_Kernel[filter_Rad + t] * temp *2.0f;
sum_2 += c_Kernel[filter_Rad + t - 1] * temp *2.0f;

}

sum_2 += c_Kernel[2 * filter_Rad] * 2.0f * s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X + filter_Rad / 2 + 1];

d_Dst[2 * (i - Halo_steps)* ROWS_BLOCKDIM_X] = sum_1;
if (pos_x + 1 < n_imageW) d_Dst[2 * (i - Halo_steps) * ROWS_BLOCKDIM_X + 1] = sum_2;
}
}
}
}