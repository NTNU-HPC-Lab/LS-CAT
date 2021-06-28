#include "includes.h"

using namespace std;

// parameter describing the size of matrix A
const int rows = 4096;
const int cols = 4096;

const int BLOCK_SIZE = 16;

// transpose shared kernel

// transpose kernel

__global__ void transpose_shared(float* a, float*b) {
__shared__ float result[BLOCK_SIZE][BLOCK_SIZE+1];
// adding one to avoid bank conflict

int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int width = gridDim.x * blockDim.x;
int height = gridDim.y * blockDim.y;

// perform transpose
if (x < height && y < width) {
result[threadIdx.x][threadIdx.y] = a[y*height + x];
}
__syncthreads();
b[x*height + y] = result[threadIdx.x][threadIdx.y];
}