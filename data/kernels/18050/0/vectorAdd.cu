#include "includes.h"
/*-----This is a vector addition--*/
/*---- @ Cuda/c ------*/
/*---- __NS__Bologna__2020__*/




__global__ void vectorAdd(int* a, int* b, int* c, int n){
// calculate index thread
int tid = blockIdx.x * blockDim.x + threadIdx.x;
// Make sure we stay in-bounds
if(tid < n)
// Vector add
c[tid] = a[tid] + b[tid];
}