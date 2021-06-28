#include "includes.h"
//
// Created by Sowmya Parameshwara on 11/10/16.
//

/**
*
*  1) Input is stored by transposing the matrix, so that the attributes of a column are stored in a single row. This
*      will optimise the algorithm since all threads in a block will access nearby elements, while normalising.
*  2) Each row is normalised at a time for calculating standardscore, the calculated values are stored in output matrix by transposing.
*  3) Number of threads in a block is set as 16 (This value determined by checking performance for different values). The number of blocks
*     is decided based on matrix size "N" and number of threads.
*  4) The contents of a row are divided among the blocks. In each block,Each thread populates one elements of the block into shared data.
*     We then calculate partial sum without divergence, on the data stored in shared memory.
*  5) Once all blocks compute partial sum, we launch a kernel function on a single block by passing the calculated values from the previous step.
*     This will calculate the final sum and final squared sum. To this final block we ensure the size of the partial sum array passed equals
the next nearest power of 2 of "the number of blocks", as partial sum algorithm works only for powers of 2.
*  6)  The above data is used to calculate standard deviation for that row using the formula ((totalSquareSum + N*powf(mean, 2.0) - 2 * mean * totalSum)/(float)N)
*  7)  The above value is used to calculate standard score for every element in that row.
*  8)  The above step repeats for every row, calculating the standard score for all elements in the row.
*
*  Steps to compile and execute on Jarvis :
*  1)  qlogin -q interactive.q  (Launches interactive session).
*  2)  nvcc matrixNorm.cu -o matrixNorm (Compile code on jarvis).
*  3)  cd hw4 (Code is available here).
*  4) ./matrixNorm 15000 4   <Argument 1 : Size of matrix, Argument 2 : Random seed value>
*/


/* Program Parameters */
#define MAXN 15000  /* Max value of N */
int N;  /* Matrix size */

/* Matrices */
volatile float A[MAXN][MAXN], B[MAXN][MAXN];

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void matrixNorm();

/* returns a seed for srand based on the time */
__global__ void block_sum(const float *hostInput, float *sumResults, float *squareResults, const size_t n)
{
__shared__ float sharedSumData[1024];
__shared__ float sharedSquareData[1024];

int i = blockIdx.x * blockDim.x + threadIdx.x;
int tx = threadIdx.x;
float x = 0;
if(i < n) {
x = hostInput[i];
}
sharedSumData[tx] = x;
sharedSquareData[tx] = x*x;
__syncthreads();

// block-wide reduction in _shared_ mem
for(int offset = blockDim.x / 2;
offset > 0;
offset >>= 1)
{
if(tx < offset)
{
sharedSumData[tx] += sharedSumData[tx + offset];
sharedSquareData[tx] += sharedSquareData[tx + offset];
}
__syncthreads();
}

// finally, thread 0 writes the calculated result of this block
if(threadIdx.x == 0)
{
// note that the result is per-block
// not per-thread
sumResults[blockIdx.x] = sharedSumData[0];
squareResults[blockIdx.x] = sharedSquareData[0];
}
}