#include "includes.h"
__global__ void dwt_per_Y_E(float *d_ip, int rows, int cols, int cA_rows, int filt_len, int Halo_steps, float *d_cL, float *d_cH) {
extern __shared__ float s_Data[];

//Offset to the upper halo edge
const int baseX = blockIdx.x * Y_BLOCKDIM_X + threadIdx.x;
const int baseY = ((blockIdx.y * 2 * Y_RESULT_STEPS) - Halo_steps) * Y_BLOCKDIM_Y + threadIdx.y;
const int baseY1 = (blockIdx.y * Y_RESULT_STEPS) * Y_BLOCKDIM_Y + threadIdx.y;

if (baseX < cols)
{
d_ip += baseY * cols + baseX;
d_cL += baseY1 * cols + baseX;
d_cH += baseY1 * cols + baseX;

//Loading data to shared memory
//Upper halo
#pragma unroll

for (int i = 0; i < Halo_steps; i++)
{
s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = (baseY + i * Y_BLOCKDIM_Y >= 0) ? d_ip[i * Y_BLOCKDIM_Y * cols] : d_ip[(i * Y_BLOCKDIM_Y * cols) + (rows*cols)];
}

//Lower halo + Main data
#pragma unroll
for (int i = Halo_steps; i < Halo_steps + 2 * Y_RESULT_STEPS + Halo_steps; i++)
{
s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = (baseY + i * Y_BLOCKDIM_Y < rows) ? d_ip[i * Y_BLOCKDIM_Y * cols] : d_ip[(i * Y_BLOCKDIM_Y * cols) - (rows*cols)];
}

__syncthreads();

//Compute and store results
#pragma unroll
for (int i = 0; i < Y_RESULT_STEPS; i++)
{
if ((baseY1 + i * Y_BLOCKDIM_Y < cA_rows)) {
int l2 = filt_len / 2;

float sum_cL = 0, sum_cH = 0;

for (int l = 0; l < filt_len; ++l)
{
sum_cL += c_lpd[l] * s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + 2 * threadIdx.y + 2 * i * Y_BLOCKDIM_Y + Halo_steps * Y_BLOCKDIM_Y + l2 - l];
sum_cH += c_hpd[l] * s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + 2 * threadIdx.y + 2 * i * Y_BLOCKDIM_Y + Halo_steps * Y_BLOCKDIM_Y + l2 - l];
}

d_cL[i * Y_BLOCKDIM_Y * cols] = sum_cL;
d_cH[i * Y_BLOCKDIM_Y * cols] = sum_cH;
}
}
}

}