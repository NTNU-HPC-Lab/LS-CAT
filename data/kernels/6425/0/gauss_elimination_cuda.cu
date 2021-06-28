#include "includes.h"
//==========================================================================================================
// A small snippet of code to solve equation of types Ax=B using Gaussian Elimniation
// Author - Anmol Gupta, Naved Ansari
// Course - EC513 - Introduction to Computer Architecture
// Boston University
//==========================================================================================================

//==========================================================================================================
// Command to compile the code
//nvcc -o GaussianElimination GaussianElimination.cu
//==========================================================================================================


// Assertion to check for errors
__global__ void gauss_elimination_cuda(float *a_d, float *b_d ,int size) {
int idx = threadIdx.x;
int idy = threadIdx.y;

__shared__ float temp[40][40];
temp[idy][idx] = a_d[(idy * (size+1)) + idx];
__syncthreads();
//cuPrintf("T idy=%d, idx=%d, temp=%f\n", idy, idx, a_d[(idy * (size+1)) + idx]);

for(int column = 0; column < size-1; column++){
if(idy > column && idx >= column){
float t = temp[column][idx] - (temp[column][column] / temp[idy][column]) * temp[idy][idx];
__syncthreads();
temp[idy][idx] = t;
}
__syncthreads();
}

b_d[idy*(size+1) + idx] = temp[idy][idx];
}