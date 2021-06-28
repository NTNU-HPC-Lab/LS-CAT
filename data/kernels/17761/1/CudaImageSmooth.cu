#include "includes.h"
__global__ void CudaImageSmooth(unsigned char *In, unsigned char *Out, int width, int height, int fsize)
{
int row, col, destIndex;

col = blockIdx.x*blockDim.x + threadIdx.x;
row = blockIdx.y*blockDim.y + threadIdx.y;
destIndex = row*width + col;
int frow, fcol;
float tmp = 0.0;

if(col < fsize/2 || col > width-fsize/2 || row < fsize-2 || row > width-fsize/2) {
Out[destIndex] = 0;
} else {
for(frow = -fsize/2; frow <= fsize/2; frow++) {
for(fcol = -fsize/2; fcol <= fsize/2; fcol++) {
tmp += (float)In[(row+frow)*width+(col+fcol)];
}
}
tmp /= (fsize*fsize);	// average
Out[destIndex] = (unsigned char)tmp;
}
}