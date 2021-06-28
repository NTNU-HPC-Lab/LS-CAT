#include "includes.h"
__global__ void setBoundaryInt2_kernel(int* d_boundary, int startPos, int numKey, int rLen, int2* d_boundaryRange)
{
const int by = blockIdx.y;
const int bx = blockIdx.x;
const int tx = threadIdx.x;
const int ty = threadIdx.y;
const int tid=tx+ty*blockDim.x;
const int bid=bx+by*gridDim.x;
const int numThread=blockDim.x;
const int resultID=(bid)*numThread+tid;
int pos=startPos+resultID;

if(pos<numKey)
{
int2 flag;
flag.x=d_boundary[pos];
if((pos+1)!=numKey)
flag.y=d_boundary[pos+1];
else
flag.y=rLen;
d_boundaryRange[pos]=flag;
}
}