#include "includes.h"
__global__ void gpu_Comput (int *h, int N, int T) {

// Array loaded with global thread ID that acesses that location

int col = threadIdx.x + blockDim.x * blockIdx.x;
int row = threadIdx.y + blockDim.y * blockIdx.y;

int threadID = col + row * N;
int index = row + col * N;		// sequentially down each row

for (int t = 0; t < T; t++)		// loop to repeat to reduce other time effects
h[index] = threadID;  		// load array with flattened global thread ID
}