#include "includes.h"

#define HISTOGRAM_LENGTH 256












__global__ void convertToChar(float * input, unsigned char * ucharInput, int width, int height)
{
int bx = blockIdx.x;  int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;

int row = by*blockDim.y+ty;
int col = bx*blockDim.x+tx;
int index = row*width + col;

if(row < height && col < width)
{
ucharInput[index*3]   = (unsigned char) (255 * input[index*3]); //r
ucharInput[index*3+1] = (unsigned char) (255 * input[index*3+1]); //g
ucharInput[index*3+2] = (unsigned char) (255 * input[index*3+2]); //b
}


}