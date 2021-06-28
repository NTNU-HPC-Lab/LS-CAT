#include "includes.h"
__global__ void SubtractMulti(float *d_Result, float *d_Data, int width, int pitch, int height)
{
const int x = blockIdx.x*SUBTRACTM_W + threadIdx.x;
const int y = blockIdx.y*SUBTRACTM_H + threadIdx.y;
int sz = height*pitch;
int p = threadIdx.z*sz + y*pitch + x;
if (x<width && y<height)
d_Result[p] = d_Data[p] - d_Data[p + sz];
__syncthreads();
}