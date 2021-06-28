#include "includes.h"
__global__ void cuImageBrighten(const float *dev_image, float *dev_out, int w, int h)
{
int tx = threadIdx.x;   int ty = threadIdx.y;
int bx = blockIdx.x;	int by = blockIdx.y;

int pos = tx + 32*bx + w* ty + 32*w*by;
dev_out[pos] = min(255.0f, dev_image[pos] + 50);
__syncthreads();
}