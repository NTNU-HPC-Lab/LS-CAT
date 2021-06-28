#include "includes.h"


#define BLOCK_DIM_X	16
#define BLOCK_DIM_Y	32


#define N	16

__global__ static void KerBitReversalMatrixCol(float *d_lpDstRe, float *d_lpDstIm, float *d_lpSrcRe, float *d_lpSrcIm, int width, int log2y)
{
register int x = blockDim.x * blockIdx.x + threadIdx.x;
register int y = blockDim.y * blockIdx.y + threadIdx.y;
//	int height = 1 << log2y;

if(y < (1 << log2y))
//	for(int i = 0; i < length; i ++)
{
register int index	= 0;
register int t	= y;

for(int j = 0; j < log2y; j ++)
{
index = (index << 1) | (t & 1);
t >>= 1;
}

if(y >= index)
{
register int idx	= width * y + x;
register int jdx	= width * index + x;

register double	 tmpRe	= d_lpDstRe[idx];
register double	 tmpIm	= d_lpDstIm[idx];

d_lpDstRe[idx]	= d_lpSrcRe[jdx];
d_lpDstIm[idx]	= d_lpSrcIm[jdx];

d_lpDstRe[jdx]	= tmpRe;
d_lpDstIm[jdx]	= tmpIm;
}
}
}