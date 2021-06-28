#include "includes.h"





__global__ void kernel_C( float * _g_data, int dimx, int dimy )
{
float2* g_data = reinterpret_cast<float2 *>(_g_data);

int id  = blockIdx.x*blockDim.x + threadIdx.x;

float2 value = g_data[id];

value.x += sqrtf( cosf(value.x) + 1.f );
value.y += sqrtf( logf(value.y) + 1.f );

g_data[id] = value;
}