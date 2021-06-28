#include "includes.h"


/*
WRITE CUDA KERNEL FOR TRANSPOSE HERE
*/
const int CHUNK_SIZE = 32;
const int CHUNK_ROWS = 8;


__global__ void matrix_t(int* data, int* out, int* rows, int* cols){
__shared__ int chunk[CHUNK_SIZE][CHUNK_SIZE];
int x = blockIdx.x * CHUNK_SIZE + threadIdx.x;
int y = blockIdx.y * CHUNK_SIZE + threadIdx.y;

for (int i=0; i<CHUNK_SIZE; i+= CHUNK_ROWS) {
chunk[threadIdx.x][threadIdx.y+i] = data[(y + i) * *cols + x];
}
__syncthreads();

x = blockIdx.y * CHUNK_SIZE + threadIdx.x;
y = blockIdx.x * CHUNK_SIZE + threadIdx.y;

for (int i=0; i<CHUNK_SIZE; i+= CHUNK_ROWS) {
if (x < *rows && y+i < *cols) {
out[(y + i) * *rows + x] = chunk[threadIdx.y + i][threadIdx.x];
//            out[(y + i) * *rows + x] = 1;
}
}
}