#include "includes.h"





__global__ void kernel_A( float *g_data, int dimx, int dimy )
{
int ix  = blockIdx.x;
int iy  = blockIdx.y*blockDim.y + threadIdx.y;
int idx = iy*dimx + ix;

float value = g_data[idx];

if( ix % 2 )
{
value += sqrtf( logf(value) + 1.f );
}
else
{
value += sqrtf( cosf(value) + 1.f );
}

g_data[idx] = value;
}