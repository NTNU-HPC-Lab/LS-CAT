#include "includes.h"

#define HISTOGRAM_LENGTH 256












__global__ void convertToGrayScale(unsigned char * ucharImg, unsigned char * grayImg, int width, int height)
{

int bx = blockIdx.x;  int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;

int row = by*blockDim.y+ty;
int col = bx*blockDim.x+tx;
int index = row*width + col;

if(row < height && col < width)
{
grayImg[index] = (unsigned char) (0.21*ucharImg[index*3] + 0.71*ucharImg[index*3 + 1] + 0.07*ucharImg[index*3 + 2]);
}

}