#include "includes.h"
__global__ void kernel_image2D1C_ConvolveRow(float* img, int n_x, int n_y, short k, float *kernel, float* out)
{
// Find index of current thread
int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
if (idx_x>=n_x) return;
if (idx_y>=n_y) return;

float sum=0;
for (short i=-k;i<=k;i++)
{
short x=idx_x+i;
if (x<0) x=0;
if (x>=n_x) x=n_x-1;
sum+=kernel[i+k]*img[idx_y*n_x+x];
}
out[idx_y*n_x+idx_x]=sum;
}