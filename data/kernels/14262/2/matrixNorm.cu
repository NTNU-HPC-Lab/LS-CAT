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
__global__ void matrixNorm(float* d_in, float* d_out, float* d_mean, float* d_sd, int N)
{
int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

unsigned int i = idx_y * N + idx_x;

if (d_sd[blockIdx.y] == 0.0)
d_out[i] = 0.0;
else
d_out[i] = (d_in[i] - d_mean[blockIdx.x]) / d_sd[blockIdx.x];
}