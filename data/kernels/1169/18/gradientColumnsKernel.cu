#include "includes.h"
__global__ void gradientColumnsKernel( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD )
{
__shared__ float s_Data[COLUMNS_GRAD_BLOCKDIM_Z][COLUMNS_GRAD_BLOCKDIM_X][(COLUMNS_GRAD_RESULT_STEPS + 2 * COLUMNS_GRAD_HALO_STEPS) * COLUMNS_GRAD_BLOCKDIM_Y + 1];

//Offset to the upper halo edge
const int baseX = blockIdx.x * COLUMNS_GRAD_BLOCKDIM_X + threadIdx.x;
const int baseY = (blockIdx.y * COLUMNS_GRAD_RESULT_STEPS - COLUMNS_GRAD_HALO_STEPS) * COLUMNS_GRAD_BLOCKDIM_Y + threadIdx.y;
const int baseZ = blockIdx.z * COLUMNS_GRAD_BLOCKDIM_Z + threadIdx.z;
d_Src += (baseZ * imageH + baseY) * imageW + baseX;
d_Dst += (baseZ * imageH + baseY) * imageW + baseX;

//Main data
#pragma unroll

for (int i = COLUMNS_GRAD_HALO_STEPS; i < COLUMNS_GRAD_HALO_STEPS + COLUMNS_GRAD_RESULT_STEPS; i++) {
s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_GRAD_BLOCKDIM_Y] = d_Src[i * COLUMNS_GRAD_BLOCKDIM_Y * imageW];
}

//Upper halo
#pragma unroll

for (int i = 0; i < COLUMNS_GRAD_HALO_STEPS; i++) {
s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_GRAD_BLOCKDIM_Y] = (baseY + i * COLUMNS_GRAD_BLOCKDIM_Y >= 0) ? d_Src[i * COLUMNS_GRAD_BLOCKDIM_Y * imageW] : 0;
}

//Lower halo
#pragma unroll

for (int i = COLUMNS_GRAD_HALO_STEPS + COLUMNS_GRAD_RESULT_STEPS; i < COLUMNS_GRAD_HALO_STEPS + COLUMNS_GRAD_RESULT_STEPS + COLUMNS_GRAD_HALO_STEPS; i++) {
s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_GRAD_BLOCKDIM_Y]= (baseY + i * COLUMNS_GRAD_BLOCKDIM_Y < imageH) ? d_Src[i * COLUMNS_GRAD_BLOCKDIM_Y * imageW] : 0;
}

//Compute and store results
__syncthreads();
#pragma unroll

for (int i = COLUMNS_GRAD_HALO_STEPS; i < COLUMNS_GRAD_HALO_STEPS + COLUMNS_GRAD_RESULT_STEPS; i++) {
float sum = 0;
sum += s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_GRAD_BLOCKDIM_Y + 1];
sum -= s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_GRAD_BLOCKDIM_Y - 1];
sum *= 0.5f;

d_Dst[i * COLUMNS_GRAD_BLOCKDIM_Y * imageW] = sum;
}
}