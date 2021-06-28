#include "includes.h"
__global__ void reduce(int *a, int *res){
// create shared memory for the threads in the block
__shared__ int cache[threadsPerBlock];

// get the thread id
int tid = threadIdx.x + blockIdx.x * blockDim.x;

// index into the cache for this block
int cacheIndex = threadIdx.x;

// set the value in cache
cache[cacheIndex] = a[tid];

__syncthreads(); //synchronize threads before continuing

int i = blockDim.x/2; // only want first half to do work
while( i != 0 ){
if (cacheIndex < i) // make sure we are not doing bogus add

// add the current index and ith element
cache[cacheIndex] += cache[cacheIndex + i];

__syncthreads(); // we want all threads to finish
i /= 2;
}
if (cacheIndex == 0) // only one thread needs to do this
*res = cache[0];
}