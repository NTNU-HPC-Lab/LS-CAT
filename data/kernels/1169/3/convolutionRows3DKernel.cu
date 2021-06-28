#include "includes.h"
__global__ void convolutionRows3DKernel( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, int kernel_index, int kernel_radius )
{
__shared__ float s_Data[ROWS_BLOCKDIM_Z][ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

//Offset to the left halo edge
const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
const int baseZ = blockIdx.z * ROWS_BLOCKDIM_Z + threadIdx.z;

d_Src += (baseZ * imageH + baseY) * imageW + baseX;
d_Dst += (baseZ * imageH + baseY) * imageW + baseX;

const float* kernel = &c_Kernel[kernel_index*MAX_KERNEL_LENGTH];

//Load main data
#pragma unroll

for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
}

//Load left halo
#pragma unroll

for (int i = 0; i < ROWS_HALO_STEPS; i++) {
s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
}

//Load right halo
#pragma unroll

for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
}

//Compute and store results
__syncthreads();
#pragma unroll

for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
{
float sum = 0;

//#pragma unroll

for (int j = -kernel_radius; j <= kernel_radius; j++)
{
sum += kernel[kernel_radius - j] * s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
}

d_Dst[i * ROWS_BLOCKDIM_X] = sum;
}
}