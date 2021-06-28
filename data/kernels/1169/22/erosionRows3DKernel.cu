#include "includes.h"
__global__ void erosionRows3DKernel ( unsigned short *d_dst, unsigned short *d_src, int w, int h, int d, int kernel_radius )
{
__shared__ unsigned short smem[ER_ROWS_BLOCKDIM_Z][ER_ROWS_BLOCKDIM_Y][(ER_ROWS_RESULT_STEPS + 2 * ER_ROWS_HALO_STEPS) * ER_ROWS_BLOCKDIM_X];
unsigned short *smem_thread = smem[threadIdx.z][threadIdx.y];

//Offset to the left halo edge
const int baseX = (blockIdx.x * ER_ROWS_RESULT_STEPS - ER_ROWS_HALO_STEPS) * ER_ROWS_BLOCKDIM_X + threadIdx.x;
const int baseY = blockIdx.y * ER_ROWS_BLOCKDIM_Y + threadIdx.y;
const int baseZ = blockIdx.z * ER_ROWS_BLOCKDIM_Z + threadIdx.z;

d_src += (baseZ * h + baseY) * w + baseX;
d_dst += (baseZ * h + baseY) * w + baseX;

//Load main data
#pragma unroll
for (int i = ER_ROWS_HALO_STEPS; i < ER_ROWS_HALO_STEPS + ER_ROWS_RESULT_STEPS; i++) {
smem_thread[threadIdx.x + i * ER_ROWS_BLOCKDIM_X] = d_src[i * ER_ROWS_BLOCKDIM_X];
}

//Load left halo
#pragma unroll
for (int i = 0; i < ER_ROWS_HALO_STEPS; i++) {
smem_thread[threadIdx.x + i * ER_ROWS_BLOCKDIM_X] = (baseX + i * ER_ROWS_BLOCKDIM_X >= 0) ? d_src[i * ER_ROWS_BLOCKDIM_X] : 0;
}

//Load right halo
#pragma unroll
for (int i = ER_ROWS_HALO_STEPS + ER_ROWS_RESULT_STEPS; i < ER_ROWS_HALO_STEPS + ER_ROWS_RESULT_STEPS + ER_ROWS_HALO_STEPS; i++) {
smem_thread[threadIdx.x + i * ER_ROWS_BLOCKDIM_X] = (baseX + i * ER_ROWS_BLOCKDIM_X < w) ? d_src[i * ER_ROWS_BLOCKDIM_X] : 0;
}

//Compute and store results
__syncthreads();
#pragma unroll
for (int i = ER_ROWS_HALO_STEPS; i < ER_ROWS_HALO_STEPS + ER_ROWS_RESULT_STEPS; i++) {
unsigned short *smem_kern = &smem_thread[threadIdx.x + i * ER_ROWS_BLOCKDIM_X - kernel_radius];
unsigned short val = smem_kern[0];

//#pragma unroll
for (int j = 1; j <= 2*kernel_radius; j++) {
val = min(val, smem_kern[j]);
}
d_dst[i * ER_ROWS_BLOCKDIM_X] = val;
}
}