#include "includes.h"

#define TILE_SIZE 26
#define RADIUS 3
#define BLOCK_SIZE (TILE_SIZE+(2*RADIUS))

texture<unsigned char, 1, cudaReadModeElementType> texInImage;
texture<unsigned int, 1, cudaReadModeElementType> texIntegralImage;

__device__ unsigned int keypointsCount = 0;






__global__ void kernel_initIntegrtalImage(unsigned int *_d_out_integralImage,  int _h_width, int _h_height)
{
int bx = blockIdx.x;
int tx = threadIdx.x;

int index = bx + tx*_h_width;
_d_out_integralImage[index] = (unsigned int)0;
}