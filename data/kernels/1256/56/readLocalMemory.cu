#include "includes.h"
__global__ void readLocalMemory(const float *data, float *output, int size, int repeat)
{
int gid = threadIdx.x + (blockDim.x * blockIdx.x), j = 0;
float sum = 0;
int tid=threadIdx.x, localSize=blockDim.x, grpid=blockIdx.x,
litems=2048/localSize, goffset=localSize*grpid+tid*litems;
int s = tid;
__shared__ float lbuf[2048];
for ( ; j<litems && j<(size-goffset) ; ++j)
lbuf[tid*litems+j] = data[goffset+j];
for (int i=0 ; j<litems ; ++j,++i)
lbuf[tid*litems+j] = data[i];
__syncthreads();
for (j=0 ; j<repeat ; ++j)
{
float a0 = lbuf[(s+0)&(2047)];
float a1 = lbuf[(s+1)&(2047)];
float a2 = lbuf[(s+2)&(2047)];
float a3 = lbuf[(s+3)&(2047)];
float a4 = lbuf[(s+4)&(2047)];
float a5 = lbuf[(s+5)&(2047)];
float a6 = lbuf[(s+6)&(2047)];
float a7 = lbuf[(s+7)&(2047)];
float a8 = lbuf[(s+8)&(2047)];
float a9 = lbuf[(s+9)&(2047)];
float a10 = lbuf[(s+10)&(2047)];
float a11 = lbuf[(s+11)&(2047)];
float a12 = lbuf[(s+12)&(2047)];
float a13 = lbuf[(s+13)&(2047)];
float a14 = lbuf[(s+14)&(2047)];
float a15 = lbuf[(s+15)&(2047)];
sum += a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15;
s = (s+16)&(2047);
}
output[gid] = sum;
}