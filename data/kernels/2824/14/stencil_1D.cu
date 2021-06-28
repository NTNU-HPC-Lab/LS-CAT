#include "includes.h"
__global__ void stencil_1D(int *in, int *out, int dim){

__shared__ int temp[BLOCKSIZE + 2*RADIUS];

int lindex = threadIdx.x + RADIUS;
int gindex = threadIdx.x + blockDim.x * blockIdx.x;
int stride = gridDim.x * blockDim.x;
int left, right;

// Go through all data
// Step all threads in a block to avoid synchronization problem
while ( gindex < (dim + blockDim.x) ) {

// Read input elements into shared memory
temp[lindex] = 0;
if (gindex < dim)
temp[lindex] = in[gindex];

// Populate halos, set to zero if we are at the boundary
if (threadIdx.x < RADIUS) {

temp[lindex - RADIUS] = 0;
left = gindex - RADIUS;
if (left >= 0)
temp[lindex - RADIUS] = in[left];

temp[lindex + blockDim.x] = 0;
right = gindex + blockDim.x;
if (right < dim)
temp[lindex + blockDim.x] = in[right];
}

// Synchronize threads - make sure all data is available!
__syncthreads();

// Apply the stencil
int result = 0;
for (int offset = -RADIUS; offset <= RADIUS; offset++) {
result += temp[lindex + offset];
}

// Store the result
if (gindex < dim)
out[gindex] = result;

// Update global index and quit if we are done
gindex += stride;

__syncthreads();

}

}