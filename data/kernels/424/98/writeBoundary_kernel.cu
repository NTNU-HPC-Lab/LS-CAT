#include "includes.h"
__global__ void writeBoundary_kernel(int startPos, int rLen, int* d_startArray, int* d_startSumArray, int* d_bounary)
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

if(pos<rLen)
{
int flag=d_startArray[pos];
int writePos=d_startSumArray[pos];
if(flag==1)
d_bounary[writePos]=pos;
}
}