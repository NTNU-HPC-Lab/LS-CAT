#include "includes.h"

#define TILE_SIZE 26
#define RADIUS 3
#define BLOCK_SIZE (TILE_SIZE+(2*RADIUS))

texture<unsigned char, 1, cudaReadModeElementType> texInImage;
texture<unsigned int, 1, cudaReadModeElementType> texIntegralImage;

__device__ unsigned int keypointsCount = 0;






__global__ void kernel_scanNaiveSumHirizontal(unsigned int *_d_out_integralImage, int _h_width, int _h_height)
{
unsigned int tmp[1024];

int tx = threadIdx.x;



for(int i = 0 ; i < _h_height; i++)
{
tmp[i] = (unsigned int )tex1Dfetch(texIntegralImage, tx + i*_h_width);
}

for(int i = 1; i < _h_height; i++)
{
tmp[i] = tmp[i-1] + tmp[i];
}

for(int i = 0 ; i < _h_height; i++)
{
_d_out_integralImage[tx + i * _h_width] = tmp[i];
}
}