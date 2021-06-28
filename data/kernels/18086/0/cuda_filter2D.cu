#include "includes.h"
__global__ void cuda_filter2D(float *dst, float *src, float *kernel, int src_width, int src_height, int kernel_rows, int kernel_cols)
{
int row = threadIdx.y + blockIdx.y * blockDim.y;
int col = threadIdx.x + blockIdx.x * blockDim.x;
if(row < src_height && col < src_width)
{
float sum = 0;
for(int i = 0; i < kernel_rows; i++)
{
for(int j = 0; j < kernel_cols; j++)
{
if(row + i - (kernel_rows - 1) / 2 >= 0 &&
col + j - (kernel_cols - 1) / 2 >= 0 &&
col + j - (kernel_cols - 1) / 2 < src_width &&
row + i - (kernel_rows - 1) / 2 < src_height)
{
sum = sum + kernel[i * kernel_cols + j] * (float)src[(row + i - (kernel_rows - 1)/2) * src_width + col +j - (kernel_rows - 1)/2];
}
}
}
dst[row *src_width + col] = (sum <0)?0:(sum>255?255:float(sum));
#ifdef debug
printf("filter2D: dst[%d] = %f\n", row * src_width + col, dst[row * src_width + col]);
#endif
}
}