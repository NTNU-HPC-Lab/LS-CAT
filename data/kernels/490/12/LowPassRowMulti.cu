#include "includes.h"
__global__ void LowPassRowMulti(float *d_Result, float *d_Data, int width, int pitch, int height)
{
__shared__ float data[CONVROW_W + 2*RADIUS];
const int tx = threadIdx.x;
const int block = blockIdx.x/(NUM_SCALES+3);
const int scale = blockIdx.x - (NUM_SCALES+3)*block;
const int xout = block*CONVROW_W + tx;
const int loadPos = xout - RADIUS;
const int yptr = blockIdx.y*pitch;
const int writePos = yptr + height*pitch*scale + xout;
float *kernel = d_Kernel + scale*16;

if (loadPos<0)
data[tx] = d_Data[yptr];
else if (loadPos>=width)
data[tx] = d_Data[yptr + width-1];
else
data[tx] = d_Data[yptr + loadPos];
__syncthreads();
if (xout<width && tx<CONVROW_W)
d_Result[writePos] =
(data[tx+0] + data[tx+8])*kernel[0] +
(data[tx+1] + data[tx+7])*kernel[1] +
(data[tx+2] + data[tx+6])*kernel[2] +
(data[tx+3] + data[tx+5])*kernel[3] +
data[tx+4]*kernel[4];
__syncthreads();
}