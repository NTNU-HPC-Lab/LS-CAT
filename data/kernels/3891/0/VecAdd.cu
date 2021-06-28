#include "includes.h"
__global__ void VecAdd(int n, const float *A, const float *B, float* C) {
//DEVICE(GPU)CODE
/********************************************************************
*
* Compute C = A + B
*   where A is a (1 * n) vector
*   where B is a (1 * n) vector
*   where C is a (1 * n) vector
*
********************************************************************/
//added for extra compute time
long long start = clock64();
long long cycles_elapsed;
do{cycles_elapsed = clock64() - start;}
while(cycles_elapsed <20000);
//end of added compute time
// INSERT KERNEL CODE HERE
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < n)
C[i] = A[i] + B[i];
}