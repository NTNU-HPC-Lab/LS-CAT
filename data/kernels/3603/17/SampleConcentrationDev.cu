#include "includes.h"
__global__	void	SampleConcentrationDev(float* concentration, const uint2*	cellStartEnd)
{
const	uint	cellid = gridDim.x*gridDim.y*threadIdx.x + blockIdx.y*gridDim.x + blockIdx.x;

uint2	cellStEnd = cellStartEnd[cellid];

concentration[cellid] = cellStEnd.y - cellStEnd.x;
}