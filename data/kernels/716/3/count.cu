#include "includes.h"
__global__ void count(int *A, int *B, int n) {

int b_id 		= blockIdx.x,
b_num 	= gridDim.x,
b_size,
b_offset,
t_id 	= threadIdx.x,
t_num 	= blockDim.x,
t_size,
t_offset,
offset;

// initialize a shared memory array to store the count for each block.
__shared__ int count[MAX_VALUE];

// set intial values to zeros. Each thread sets its own share to zero.
t_size = (t_num > MAX_VALUE ? 1 : MAX_VALUE / t_num);
offset = t_id * t_size;
for (int i = offset; i < offset + t_size && i < MAX_VALUE; ++i)
count[i] = 0;

// wait until all threads have completed the initialization process.
__syncthreads();

// accumulate the counts of each value. Each thread counts a certain portain
// of the unsorted array.
b_size = (b_num > n ? 1 : n / b_num);
b_offset = b_id * b_size;

t_size = (t_num > b_size ? 1 : b_size / t_num);

offset = b_offset + t_id * t_size;
for (int i = offset; i < offset + t_size && i < b_offset + b_size && i < n; ++i)
atomicAdd(&count[A[i]], 1);

// wait until all threads have completed the couting phase.
__syncthreads();

// copy the block count into global memory. Each thread copies its portioin to
// the global memory.
t_size = (t_num > MAX_VALUE ? 1 : MAX_VALUE / t_num);
t_offset = t_id * t_size;
offset = b_id * MAX_VALUE + t_offset;

if (offset + t_size <= (b_id + 1) * MAX_VALUE)
memcpy(&B[offset], &count[t_offset], sizeof(int) * t_size);

}