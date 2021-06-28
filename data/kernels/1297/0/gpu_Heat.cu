#include "includes.h"

// First solution with global memory

// Shared memory residual calculation
// Reduction code from CUDA Slides - Mark Harris

__global__ void gpu_Heat (float *u, float *utmp, float *residual,int N) {

// TODO: kernel computation
int sizey = N;
int j = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.y * blockDim.y + threadIdx.y;
float diff=0.0;
if( i < N-1 && j < N-1 && i > 0 && j > 0) {
utmp[i*sizey+j]= 0.25 *
(u[ i*sizey     + (j-1) ]+  // left
u[ i*sizey     + (j+1) ]+  // right
u[ (i-1)*sizey + j     ]+  // top
u[ (i+1)*sizey + j     ]); // bottom
diff = utmp[i*sizey+j] - u[i*sizey + j];
residual[i*sizey+j] = diff * diff;
}
}