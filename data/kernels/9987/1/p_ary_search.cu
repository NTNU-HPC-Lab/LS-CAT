#include "includes.h"
__device__ int get_index_to_check(int thread, int num_threads, int set_size, int offset) {

// Integer division trick to round up
return (((set_size + num_threads) / num_threads) * thread) + offset;
}
__global__ void p_ary_search(int search, int array_length, int *arr, int *ret_val)
{
const int num_threads = blockDim.x * gridDim.x;
const int thread = blockIdx.x * blockDim.x + threadIdx.x;
int set_size = array_length;

ret_val[0] = -1;
ret_val[1] = 0;

while (set_size != 0)
{
int offset = ret_val[1];

__syncthreads();

// Get the next index to check
int index_to_check = get_index_to_check(thread, num_threads, set_size, offset);

// If the index is outside the bounds of the array do not check it
if (index_to_check < array_length)
{
// If the next index is outside the bounds of the array, then set it to maximum array size
int next_index_to_check = get_index_to_check(thread + 1, num_threads, set_size, offset);
if (next_index_to_check >= array_length)
{
next_index_to_check = array_length - 1;
}

// If we're at the mid section of the array reset the offset to this index
if (search > arr[index_to_check] && (search < arr[next_index_to_check]))
{
ret_val[1] = index_to_check;
}
else if (search == arr[index_to_check])
{
// Set the return var if find it
ret_val[0] = index_to_check;
}
}

// Since this is a paralel array search divide by our total threads to get the next set size
set_size = set_size / num_threads;

// Sync up so no threads jump ahead
__syncthreads();
}
}