#include "includes.h"





__global__ void kernel_D( float * _g_data, int dimx, int dimy )
{
float4* g_data = reinterpret_cast<float4 *>(_g_data);

int id  = blockIdx.x*blockDim.x + threadIdx.x;

float4 value = g_data[id];

value.x += sqrtf( cosf(value.x) + 1.f );
value.y += sqrtf( logf(value.y) + 1.f );
value.z += sqrtf( cosf(value.z) + 1.f );
value.w += sqrtf( logf(value.w) + 1.f );

g_data[id] = value;
}