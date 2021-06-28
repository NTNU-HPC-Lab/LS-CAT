#include "includes.h"
/*****************************************************************************/
// nvcc -O1 -o bpsw bpsw.cu -lrt -lm


// Assertion to check for errors
__global__ void kernel_trialDiv (long* n, int* r) {
int bx = blockIdx.x;      // ID thread
int tx = threadIdx.x;
int i=0;

// Identify the row and column of the Pd element to work on
long memIndex = bx*TILE_WIDTH+tx;
for (i = 0; i < 256; i++)
{
//		r[memIndex] = ((n[memIndex])%(d_sPrimes[i]) == 0)? (r[memIndex] - 1) : r[memIndex];			//ternary is slower than if statement
if (n[memIndex] % d_sPrimes[i] == 0)
r[memIndex] = r[memIndex] - 1;															//r decreases from 1. Only 1s are prime candidates
}

__syncthreads();
}