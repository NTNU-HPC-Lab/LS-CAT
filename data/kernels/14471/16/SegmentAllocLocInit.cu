#include "includes.h"
__global__ void SegmentAllocLocInit(ushort2* gSegments, const uint32_t segmentCount)
{
unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
if(globalId >= segmentCount) return;
gSegments[globalId].x = 0xFFFF;
gSegments[globalId].y = 0xFFFF;
}