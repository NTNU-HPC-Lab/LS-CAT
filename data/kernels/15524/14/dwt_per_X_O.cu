#include "includes.h"
__global__ void dwt_per_X_O(float *d_ip, int rows, int cols, int cA_cols, int filt_len, int Halo_steps, float *d_cL, float *d_cH)
{
extern __shared__ float s_Data[];

//Offset to the left halo edge
const int baseX = (blockIdx.x * 2 * X_RESULT_STEPS - Halo_steps) * X_BLOCKDIM_X + threadIdx.x;
const int baseX1 = (blockIdx.x * X_RESULT_STEPS) * X_BLOCKDIM_X + threadIdx.x;
const int baseY = blockIdx.y * X_BLOCKDIM_Y + threadIdx.y;

if (baseY < rows) {

d_ip += baseY * cols + baseX;
d_cL += baseY * cA_cols + baseX1;
d_cH += baseY * cA_cols + baseX1;

//Loading data to shared memory

//Load Left Halo
#pragma unroll
for (int i = 0; i < Halo_steps; i++)
{
if (baseX + i * X_BLOCKDIM_X == -1) s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = d_ip[cols - 1];

else s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = (baseX + i * X_BLOCKDIM_X >= 0) ? d_ip[i * X_BLOCKDIM_X] : d_ip[i * X_BLOCKDIM_X + cols + 1];
}

// main data and Load right halo
#pragma unroll

for (int i = Halo_steps; i < Halo_steps + 2 * X_RESULT_STEPS + Halo_steps; i++)
{
if (baseX + i * X_BLOCKDIM_X == cols) s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = d_ip[i * X_BLOCKDIM_X - 1];

else s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = ((baseX + i * X_BLOCKDIM_X) < cols) ? d_ip[i * X_BLOCKDIM_X] : d_ip[i * X_BLOCKDIM_X - cols - 1];
}

//Compute and store results
__syncthreads();


#pragma unroll

for (int i = 0; i < X_RESULT_STEPS; i++)
{
if ((baseX1 + i * X_BLOCKDIM_X < cA_cols))
{
float sum_cL = 0, sum_cH = 0;

int l2 = filt_len / 2;

for (int l = 0; l < filt_len; ++l)
{
sum_cL += c_lpd[l] * s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x * 2 + Halo_steps*X_BLOCKDIM_X + 2 * i * X_BLOCKDIM_X + l2 - l]; //l2-l is to select the right center pixels with odd and even sized filters
sum_cH += c_hpd[l] * s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x * 2 + Halo_steps*X_BLOCKDIM_X + 2 * i * X_BLOCKDIM_X + l2 - l];
}
d_cL[i * X_BLOCKDIM_X] = sum_cL;
d_cH[i * X_BLOCKDIM_X] = sum_cH;
}
}
}
}