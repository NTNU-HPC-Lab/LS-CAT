#include "includes.h"
__global__ void fill_count(float * prp_0,int sz)
{
// Thread index
int tx = threadIdx.x + blockIdx.x * blockDim.x;
int ty = threadIdx.y + blockIdx.y * blockDim.y;
int tz = threadIdx.z + blockIdx.z * blockDim.z;

prp_0[tz*sz*sz + ty*sz + tx] = tz*sz*sz + ty*sz + tx;
}