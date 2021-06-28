#include "includes.h"
__global__ void matrix_count(int* data, int* count, int* rows, int* cols){
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
if (chunk[threadIdx.y + i][threadIdx.x] == 1)
atomicAdd(count, 1);
}
}
}