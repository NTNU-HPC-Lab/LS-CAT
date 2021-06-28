#include "includes.h"

//VERSION 0.8 MODIFIED 10/25/16 12:34 by Jack

// The number of threads per blocks in the kernel
// (if we define it here, then we can use its value in the kernel,
//  for example to statically declare an array in shared memory)
const int threads_per_block = 256;


// Forward function declarations
float GPU_vector_max(float *A, int N, int kernel_code, float *kernel_time, float *transfer_time);
float CPU_vector_max(float *A, int N);
float *get_random_vector(int N);
float *get_increasing_vector(int N);
float usToSec(long long time);
long long start_timer();
long long stop_timer(long long start_time, const char *name);
void die(const char *message);
void checkError();

// Main program
__global__ void vector_max_kernel(float *in, float *out, int N) {

// Determine the "flattened" block id and thread id
int block_id = blockIdx.x + gridDim.x * blockIdx.y;
int thread_id = blockDim.x * block_id + threadIdx.x;

// A single "lead" thread in each block finds the maximum value over a range of size threads_per_block
float max = 0.0;
if (threadIdx.x == 0) {

//calculate out of bounds guard
//our block size will be 256, but our vector may not be a multiple of 256!
int end = threads_per_block;
if(thread_id + threads_per_block > N)
end = N - thread_id;

//grab the lead thread's value
max = in[thread_id];

//grab values from all other threads' locations
for(int i = 1; i < end; i++) {

//if larger, replace
if(max < in[thread_id + i])
max = in[thread_id + i];
}

out[block_id] = max;

}
}