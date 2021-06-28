#include "includes.h"
__global__ void tile_MatrixMul(int* a, int* b, int* c, int n, int tile_size) {
//statically-sized memory
__shared__ int A[Shared_Mem_Size];
__shared__ int B[Shared_Mem_Size];

int tx = threadIdx.x;
int ty = threadIdx.y;
int bx = blockIdx.x;
int by = blockIdx.y;

//cal global row and col postions for this thread
int row = by * tile_size + ty;
int col = bx * tile_size + tx;

//Intermidiate sum for element being written
int temp_val = 0;

//sweet tiles over entire matrix
for (int i = 0; i < (n / tile_size); i++)
{
/*

Every thread in a threadblock loads one element into shared memory
The element location in shared memory corresponds to the thread's
position in the threadblock (e.g thread[0,0] loads for
A[0 * tile_size + 0] and B[0 * tile_size + 0])

Explanation of indexing parameters
for A:
row*n: Indexes the global row for this thread (loop invariant)
i*tile_size: Indexes new set of column each iteration
tx: Indexes the column within that set

for B:
col: Indexes the global column this thread (loop invariant)
i*tile_size*n: Indexes next set of rows each iteration
ty*n: Indexes the row within that set
*/
A[(ty * tile_size) + tx] = a[row * n + (i * tile_size + tx)];
B[(ty * tile_size) + tx] = b[(i * tile_size * n + ty * n) + col];

//Ensure all threads have loaded their data before proceeding
__syncthreads();

//cal all temp values for this tile
for (int j = 0; j < tile_size; j++)
{
temp_val += A[(ty * tile_size) + j] * B[(j * tile_size) + tx];
}

//Ensure some threads dont progress and stomp current shared memory values
__syncthreads();
}
c[(row * n) + col] = temp_val;
}