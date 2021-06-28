#include "includes.h"
__global__ void idwt_per_Y_1(float *d_dst, float *src_A, float *src_D, int rows, int cols, int next_rows, int filt_len, int halo) {

extern __shared__ float s_Data[];

//Offset to the upper halo edge
const int baseX = blockIdx.x * I_Y_BLOCKDIM_X + threadIdx.x;
const int baseY = ((blockIdx.y * I_Y_RESULT_STEPS) - halo) * I_Y_BLOCKDIM_Y + threadIdx.y;

int l2 = filt_len / 2;

if (baseX < cols)
{
src_A += baseY * cols + baseX;
src_D += baseY * cols + baseX;
d_dst += (2 * baseY - l2 + 1) * cols + baseX;

//Loading data to shared memory
//Upper halo
#pragma unroll

for (int i = 0; i < halo; i++)
{
s_Data[(threadIdx.x*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y] = (baseY + i * I_Y_BLOCKDIM_Y >= 0) ? src_A[i * I_Y_BLOCKDIM_Y * cols] : src_A[(i * I_Y_BLOCKDIM_Y * cols) + (rows*cols)];
s_Data[((threadIdx.x + I_Y_BLOCKDIM_X)*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y] = (baseY + i * I_Y_BLOCKDIM_Y >= 0) ? src_D[i * I_Y_BLOCKDIM_Y * cols] : src_D[(i * I_Y_BLOCKDIM_Y * cols) + (rows*cols)];
}

//Lower halo + Main data
#pragma unroll
for (int i = halo; i < halo + I_Y_RESULT_STEPS + halo; i++)
{
s_Data[(threadIdx.x*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y] = (baseY + i * I_Y_BLOCKDIM_Y < rows) ? src_A[i * I_Y_BLOCKDIM_Y * cols] : src_A[(i * I_Y_BLOCKDIM_Y * cols) - (rows*cols)];
s_Data[((threadIdx.x + I_Y_BLOCKDIM_X)*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y] = (baseY + i * I_Y_BLOCKDIM_Y < rows) ? src_D[i * I_Y_BLOCKDIM_Y * cols] : src_D[(i * I_Y_BLOCKDIM_Y * cols) - (rows*cols)];
}

__syncthreads();
#pragma unroll
for (int i = halo; i < I_Y_RESULT_STEPS + halo; i++)
{
int pos_y = 2 * baseY + 2 * i * I_Y_BLOCKDIM_Y;

if (pos_y + 1 < (2 * rows + filt_len - 2)) {

float temp_1 = 0, temp_2 = 0;


for (int l = 0; l < l2; ++l)
{
int t = 2 * l;

temp_1 += c_lpr[t] * s_Data[(threadIdx.x*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i* I_Y_BLOCKDIM_Y - l]
+ c_hpr[t] * s_Data[((threadIdx.x + I_Y_BLOCKDIM_X)*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y - l];
temp_2 += c_lpr[t + 1] * s_Data[(threadIdx.x*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y - l]
+ c_hpr[t + 1] * s_Data[((threadIdx.x + I_Y_BLOCKDIM_X)*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y - l];
}

if ((pos_y >= l2 - 1) && (pos_y < next_rows + l2 - 1)) d_dst[2 * i * I_Y_BLOCKDIM_Y * cols] = temp_1;
if ((pos_y + 1 >= l2 - 1) && (pos_y + 1 < next_rows + l2 - 1)) d_dst[(2 * i * I_Y_BLOCKDIM_Y + 1) * cols] = temp_2;
}
}

}
}