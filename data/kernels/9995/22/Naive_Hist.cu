#include "includes.h"
__global__ void Naive_Hist(int* d_result, int* d_hist, int n_vertices) {

//each block compares the same row to all others row2
int row = blockIdx.x;
int row2 = threadIdx.x;
bool equal;

//shared count for whole block/same vertice
__shared__ int count;

//one thread sets count to zero and syncsthreads.
if(row2 == 0)
count = 0;
__syncthreads();

//checks equality to other vertices
if(row < n_vertices && row2 < n_vertices)
for(int i = row2; i < n_vertices; i += blockDim.x) {

//checks equality of vertices lcm
equal = false;
for(int j = 0; j < n_vertices; j++) {

if(d_result[row*n_vertices +j] == d_result[i*n_vertices + j])
equal = true;
else {
equal = false;
break;
}
}

//adds to count if vertices are equal
if(equal)
atomicAdd(&count, 1);
}

//syncsthreads so count is done and increments hist[count]
__syncthreads();
if(row < n_vertices && row2 == 0 && count > 0)
atomicAdd(&d_hist[count], 1);
}