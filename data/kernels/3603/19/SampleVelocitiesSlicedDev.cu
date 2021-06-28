#include "includes.h"
__device__ float length(float3 vec)
{
return	sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}
__device__ float length4(float4 vec)
{
return	sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}
__global__	void	SampleVelocitiesSlicedDev(float* velocities, const uint slice, const float4* vels_data, const uint2*	cellStartEnd,const uint* indices)
{
const	uint	cellid = gridDim.x*blockDim.x*slice + threadIdx.x*gridDim.x + blockIdx.x;

uint2	cellStEnd = cellStartEnd[cellid];

const uint	part_in_cell = cellStEnd.y - cellStEnd.x;

if(part_in_cell <= 0)
{
velocities[threadIdx.x*gridDim.x + blockIdx.x] = 0;
return;
}

float4	vel,p = make_float4(0,0,0,0);

for(uint	index = cellStEnd.x; index < cellStEnd.y; index++)
{
#ifndef	REORDER
uint	idx = indices[index];
vel = vels_data[idx];
#else
vel = vels_data[index];
#endif

p.x += vel.x;
p.y += vel.y;
p.z += vel.z;
}

velocities[threadIdx.x*gridDim.x + blockIdx.x] = length4(p) / part_in_cell;
}