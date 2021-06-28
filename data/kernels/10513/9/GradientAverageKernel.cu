#include "includes.h"
__global__ void GradientAverageKernel(float4 *D, float4 *TD, unsigned int *NEIGHBOR, unsigned int *NBOFFSETS, unsigned int *nNeighbors, unsigned int nVertices)
{
int n,N;
int offset,soffset;

// since we are using multiple threads per blocks as well as multiple blocks
int vidxb = 4*(blockIdx.x * blockDim.x) + threadIdx.x;
//int basevert = 4*(blockIdx.x * blockDim.x);

int vidx; //,tab;
float4 nbd,td;

// create a cache for 4 elements per block (4*BLOCK_SIZE elements)
__shared__ float4 SI[4*BLOCK_SIZE_AVGG];

int bidx = 4*threadIdx.x;
// this means we have 128 neighboring vertices cached
for (vidx=vidxb; vidx<vidxb+4*BLOCK_SIZE_AVGG; vidx+=BLOCK_SIZE_AVGG)
{
if (vidx < nVertices)
{
SI[bidx] = D[vidx];
bidx++;
}
}

__syncthreads();

bidx = 4*threadIdx.x;
// preload the current BLOCK_SIZE vertices
for (vidx=vidxb; vidx<vidxb+4*BLOCK_SIZE_AVGG; vidx+=BLOCK_SIZE_AVGG)
{
if (vidx < nVertices)
{

offset = NBOFFSETS[ vidx ];
N = nNeighbors[ vidx ];

td = SI[bidx++];

for (n = 0; n < N; n++)
{
soffset = NEIGHBOR[offset+n];
/*
tab = soffset - basevert;
if(tab > 0 && tab < 4*BLOCK_SIZE)
nbd = SI[tab];
else
*/
nbd = D[soffset];

td.x += nbd.x;
td.y += nbd.y;
td.z += nbd.z;
}

td.x /= (float)(N+1);
td.y /= (float)(N+1);
td.z /= (float)(N+1);

TD[vidx] = td;
}
}
}