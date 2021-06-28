#include "includes.h"

/* Program Parameters */
#define MAXN 8000  /* Max value of N */
int N;  /* Matrix size */

// Thread block size
#define BLOCK_SIZE 16

/* Matrices */
float A[MAXN][MAXN], B[MAXN][MAXN];

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
/* ------------------ Cuda Code --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][] and B[][],
* defined in the beginning of this code.  B[][] is initialized to zeros.
*/




/* returns a seed for srand based on the time */
__global__ void matrixMean(float* d_in, float* d_mean, int N)
{
extern __shared__ float sdata[];

//each thread loads one element from global to shared mem
int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

unsigned int tid = threadIdx.y;
unsigned int i = idx_y * N + idx_x;
sdata[tid] = d_in[i];
__syncthreads();

// do reduction in shared mem
for(unsigned int s=1; s < blockDim.y; s *= 2)
{
if(tid +s < N)
{
if(tid % (2*s) == 0)
{
sdata[tid] += sdata[tid + s];
}
}
__syncthreads();
}

// write result for this block to global mem
if(tid == 0)
{
d_mean[blockIdx.x] = sdata[0]/(float) N;
}
}