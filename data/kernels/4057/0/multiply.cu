#include "includes.h"
#define BLOCK_SIZE 16

/*
* prints matrices
* Because matrices filled with dummy 0s function takes 3 dim arguments:
*      actual x and y dimension and dim as big square matrix's dimension
*/
__global__ void multiply(float *left, float *right, float *res, int dim) {

int i,j;
float temp = 0;

__shared__ float Left_shared_t [BLOCK_SIZE][BLOCK_SIZE];
__shared__ float Right_shared_t[BLOCK_SIZE][BLOCK_SIZE];

// Row i of matrix left
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;


for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {

// Column j of matrix left
j = tileNUM * BLOCK_SIZE + threadIdx.x;
i = tileNUM * BLOCK_SIZE + threadIdx.y;
// Load left[i][j] to shared mem

Left_shared_t[threadIdx.y][threadIdx.x] = left[row * dim + j];// Coalesced access
// Load right[i][j] to shared mem

Right_shared_t[threadIdx.y][threadIdx.x] = right[i * dim + col]; // Coalesced access
// Synchronize before computation
__syncthreads();

// Accumulate one tile of res from tiles of left and right in shared mem
for (int k = 0; k < BLOCK_SIZE; k++) {

temp += Left_shared_t[threadIdx.y][k] * Right_shared_t[k][threadIdx.x]; //no shared memory bank conflict
}
// Synchronize
__syncthreads();
}
// Store accumulated value to res
res[row * dim + col] = temp;
}