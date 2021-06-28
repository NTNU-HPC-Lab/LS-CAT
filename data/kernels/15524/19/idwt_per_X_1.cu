#include "includes.h"
__global__ void idwt_per_X_1(float *d_dst, float *src_A, float *src_D, int rows, int cols, int next_cols, int filt_len, int halo) {

extern __shared__ float s_Data[];

//Offset to the left halo edge
const int baseX = ((blockIdx.x * I_X_RESULT_STEPS) - halo) * I_X_BLOCKDIM_X + threadIdx.x; // even if the last pixel of cols+l2-1 takes a new block, rows can be maintained
const int baseY = blockIdx.y * I_X_BLOCKDIM_Y + threadIdx.y;

const int baseX1 = blockIdx.x * I_X_RESULT_STEPS * 2 * I_X_BLOCKDIM_X + 2 * threadIdx.x;

if (baseY < rows) {

src_A += baseY * cols + baseX;
src_D += baseY * cols + baseX;
d_dst += baseY * next_cols + baseX1;//To compensate the halo

int l2 = filt_len / 2;

//Loading data to shared memory
#pragma unroll
for (int i = 0; i < halo; i++)
{
s_Data[(threadIdx.y*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X] = (baseX + i * I_X_BLOCKDIM_X >= 0) ? src_A[i * I_X_BLOCKDIM_X] : src_A[i * I_X_BLOCKDIM_X + cols];
s_Data[((threadIdx.y + I_X_BLOCKDIM_Y)*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X] = (baseX + i * I_X_BLOCKDIM_X >= 0) ? src_D[i * I_X_BLOCKDIM_X] : src_D[i * I_X_BLOCKDIM_X + cols];
}

// main data and Load right halo
#pragma unroll

for (int i = halo; i < halo + I_X_RESULT_STEPS + halo; i++)
{

s_Data[(threadIdx.y*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X] = ((baseX + i * I_X_BLOCKDIM_X) < cols) ? src_A[i * I_X_BLOCKDIM_X] : src_A[i * I_X_BLOCKDIM_X - cols];
s_Data[((threadIdx.y + I_X_BLOCKDIM_Y)*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X] = ((baseX + i * I_X_BLOCKDIM_X) < cols) ? src_D[i * I_X_BLOCKDIM_X] : src_D[i * I_X_BLOCKDIM_X - cols];

}

//Compute and store results
__syncthreads();

#pragma unroll
for (int i = halo; i < halo + I_X_RESULT_STEPS; i++)
{
int pos_x = (baseX1 + 2 * (i - halo) * I_X_BLOCKDIM_X);

if ((pos_x + 1) < (2 * cols + filt_len - 2)) {

float temp_1 = 0, temp_2 = 0;

for (int l = 0; l < l2; ++l)
{
int t = 2 * l;

temp_1 += c_lpr[t] * s_Data[(threadIdx.y*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X - l]
+ c_hpr[t] * s_Data[((threadIdx.y + I_X_BLOCKDIM_Y)*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X - l];
temp_2 += c_lpr[t + 1] * s_Data[(threadIdx.y*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X - l]
+ c_hpr[t + 1] * s_Data[((threadIdx.y + I_X_BLOCKDIM_Y)*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X - l];
}

if ((pos_x >= l2 - 1) && (pos_x < (next_cols + l2 - 1))) d_dst[2 * (i - halo) * I_X_BLOCKDIM_X - l2 + 1] = temp_1;
if ((pos_x + 1 >= l2 - 1) && (pos_x + 1 < (next_cols + l2 - 1))) d_dst[2 * (i - halo) * I_X_BLOCKDIM_X - l2 + 2] = temp_2;
}

}
}
}