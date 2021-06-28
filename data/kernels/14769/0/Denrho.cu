#include "includes.h"

const int BLOCK_SIZE_X = 26;
const int BLOCK_SIZE_Y = 26;
const float w1 = 4.0/9.0, w2 = 1.0/9.0, w3 = 1.0/36.0;
const float Amp2 = 0.1, Width = 10, omega = 1;





__global__ void Denrho(float* u_d, float* f_d, int ArraySizeX, int ArraySizeY)
{
int tx = threadIdx.x;
int ty = threadIdx.y;
int bx = blockIdx.x*(BLOCK_SIZE_X-2);
int by = blockIdx.y*(BLOCK_SIZE_Y-2);
int x = tx + bx;
int y = ty + by;
u_d[x*ArraySizeY+y] = 0;
for (int i=0;i<9;i++)
u_d[x*ArraySizeY+y] += (float)f_d[x*ArraySizeY*9+y*9+i];

__syncthreads();
}