#include "includes.h"
__global__ void computeSphereVertexDistancesKernel(float4 *V, float *dist, unsigned int *NEIGHBOR, unsigned int *NBOFFSETS, unsigned int *nNeighbors, unsigned int nVertices, float circumference)
{
int n,N;
int offset,soffset;

// since we are using multiple threads per blocks as well as multiple blocks
int vidxb = 4*(blockIdx.x * blockDim.x) + threadIdx.x;
int basevert = 4*(blockIdx.x * blockDim.x);

int vidx,tab;
float4 nv,tv;
float dot,n1,n2,norm;

// create a cache for 4 elements per block (4*BLOCK_SIZE elements)
__shared__ float4 SI[4*BLOCK_SIZE_CVD];

int bidx = threadIdx.x;
// this means we have 128 neighboring vertices cached
for (vidx=vidxb; vidx<vidxb+4*BLOCK_SIZE_CVD; vidx+=BLOCK_SIZE_CVD)
{
if (vidx < nVertices)
{
SI[bidx] = V[vidx];
bidx+=BLOCK_SIZE_CVD;
}
}

__syncthreads();

bidx = threadIdx.x;
// preload the current BLOCK_SIZE vertices
for (vidx=vidxb; vidx<vidxb+4*BLOCK_SIZE_CVD; vidx+=BLOCK_SIZE_CVD)
{
if (vidx < nVertices)
{
offset = NBOFFSETS[ vidx ];
N = nNeighbors[ vidx ];
tv = SI[bidx];

bidx += BLOCK_SIZE_CVD;

for (n = 0; n < N; n++)
{
soffset = NEIGHBOR[offset+n];

/* There seems to be little to NO benefit of this local caching,
either because we have no hits, or reading from the shared memory
is just as slow as reading from global memory
*/
tab = soffset - basevert;
if (tab > 0 && tab < 4*BLOCK_SIZE_CVD)
{
nv = SI[tab];
}
else
{
nv = V[soffset];
}

// avoid FMADS
//dot = tv.x*nv.x + tv.y*nv.y + tv.z*nv.z;

dot = __fmul_rn(tv.x,nv.x);
dot = __fadd_rn(dot,__fmul_rn(tv.y,nv.y));
dot = __fadd_rn(dot,__fmul_rn(tv.z,nv.z));

//n1 = tv.x*tv.x + tv.y*tv.y + tv.z*tv.z;

n1 = __fmul_rn(tv.x,tv.x);
n1 = __fadd_rn(n1,__fmul_rn(tv.y,tv.y));
n1 = __fadd_rn(n1,__fmul_rn(tv.z,tv.z));

//n2 = nv.x*nv.x + nv.y*nv.y + nv.z*nv.z;

n2 = __fmul_rn(nv.x,nv.x);
n2 = __fadd_rn(n2,__fmul_rn(nv.y,nv.y));
n2 = __fadd_rn(n2,__fmul_rn(nv.z,nv.z));

norm = __fmul_rn(__fsqrt_rn(n1),__fsqrt_rn(n2));

//norm = __fsqrt_rn(n1) * __fsqrt_rn(n2);

// this seems to be a quell of numerical error here
if (norm < 1.0e-7f)
{
dist[offset+n] = 0.0f;
}
else if (fabsf(dot) > norm)
{
dist[offset+n] = 0.0f;
}
else
{
dist[offset+n] = __fmul_rn(circumference,fabsf(acosf(dot/norm)));
}
}
}
}
}