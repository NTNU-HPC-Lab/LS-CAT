#include "includes.h"
__global__ void copyChunks_kernel(void *d_source, int startPos, int2* d_Rin, int rLen, int *d_sum, void *d_dest)
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
int2 value=d_Rin[pos];
int offset=value.x;
int size=value.y;
int startWritePos=d_sum[pos];
int i=0;
char *source=(char*)d_source;
char *dest=(char*)d_dest;
for(i=0;i<size;i++)
{
dest[i+startWritePos]=source[i+offset];
}
value.x=startWritePos;
d_Rin[pos]=value;
}
}