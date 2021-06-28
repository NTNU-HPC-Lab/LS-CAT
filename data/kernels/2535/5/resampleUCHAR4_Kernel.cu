#include "includes.h"



#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)




__global__ void resampleUCHAR4_Kernel(uchar4* d_output, unsigned int outputWidth, unsigned int outputHeight, const uchar4* d_input, unsigned int inputWidth, unsigned int inputHeight)
{
const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

if (x < outputWidth && y < outputHeight)
{
const float scaleWidth = (float)(inputWidth-1) / (float)(outputWidth-1);
const float scaleHeight = (float)(inputHeight-1) / (float)(outputHeight-1);

const unsigned int xInput = (unsigned int)(x*scaleWidth + 0.5f);
const unsigned int yInput = (unsigned int)(y*scaleHeight + 0.5f);

if (xInput < inputWidth && yInput < inputHeight) {
d_output[y*outputWidth + x] = d_input[yInput*inputWidth + xInput];
}
}
}