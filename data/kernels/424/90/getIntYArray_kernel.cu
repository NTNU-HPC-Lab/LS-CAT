#include "includes.h"
__global__ void getIntYArray_kernel(int2* d_input, int startPos, int rLen, int* d_output)
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
int2 value=d_input[pos];
d_output[pos]=value.y;
}
}