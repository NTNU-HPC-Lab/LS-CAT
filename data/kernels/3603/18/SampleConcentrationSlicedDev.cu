#include "includes.h"
__global__	void	SampleConcentrationSlicedDev(float* concentration, uint slice,  const uint2*	cellStartEnd)
{
const	uint	cellid = gridDim.x*blockDim.x*slice + threadIdx.x*gridDim.x + blockIdx.x;

uint2	cellStEnd = cellStartEnd[cellid];

concentration[threadIdx.x*gridDim.x + blockIdx.x] = cellStEnd.y - cellStEnd.x;
}