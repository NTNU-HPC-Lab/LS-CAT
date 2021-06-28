#include "includes.h"





__global__ void kernel_B( float *g_data, int dimx, int dimy )
{
int id  = blockIdx.x*blockDim.x + threadIdx.x;

float value = g_data[id];

if( id % 2 )
{
value += sqrtf( logf(value) + 1.f );
}
else
{
value += sqrtf( cosf(value) + 1.f );
}

g_data[id] = value;
}