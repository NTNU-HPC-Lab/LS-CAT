#include "includes.h"
__global__ void cuMultOpti( int *a, int *b, int *c, int wA, int wB, int hA)
{
#define blockTile 16
/* Blocksize is 16x16 */
/* Allocate shared memory */
__shared__ int aBlock[blockTile][blockTile];
__shared__ int bBlock[blockTile][blockTile];

/* Calculate global index X, Y*/
int gidx = blockDim.x * blockIdx.x + threadIdx.x;  // column
int gidy = blockDim.y * blockIdx.y + threadIdx.y;   // row

/* Assign shared memory and sync  */
/* Warning, wA*gidy may be out of bounds */
aBlock[threadIdx.x][threadIdx.y] = a[gidy*wA + threadIdx.x];
bBlock[threadIdx.x][threadIdx.y] = b[threadIdx.y*wB + gidx];

/* Make sure all of the threads have cached the memory */
__syncthreads();

/* Check if global IDs are within limits */
if(gidx < wB && gidy < hA)
{
int sum = 0;
for(int k=0; k<wA; k++)
{
sum += aBlock[threadIdx.y][k] * bBlock[k][threadIdx.x];
}
// c [gidy][gidx]
c[gidy * wB + gidx] = sum;

}
}