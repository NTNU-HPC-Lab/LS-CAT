#include "includes.h"
//Author: Adriel Kim
//6-27-2020
//Updated 7-5-2020
/*
Desc: Basic 2D matrix operations - element-wise addition, subtraction, multiplication, and division.

To do:
- Use vector instead of array?
- Be able to test for varying sizes of images. (For now we manually define with constant N)
- Add timer to compare CPU and GPU implementations
- Double check if all memory is freed
- Optimize by eliminating redundant calculations
- Test code on department servers
*/


//define imin(a,b)  (a<b?a:b)//example of ternary operator in c++
//4176,2048
#define R 4176
#define C 2048
#define N (R*C)//# of elements in matrices
const int threadsPerBlock = 1024;//threads in a block. A chunk that shares the same shared memory.
const int blocksPerGrid = 8352;//imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);//this will be our output array size for sumKernel.

using namespace std;

cudaError_t matrixOperation(double* c, const double* a, const double* b, unsigned int arrSize, int operation, float* kernel_runtime, float* GPU_transfer_time);
void CPUMatrixOperation(double* c, const double* a, const double* b, unsigned int arrSize, int operation);
long long start_timer();
long long stop_timer(long long start_time, const char *name);

//any advantages with mapping directly to strucutre of matrix? We're just representing 2D matrix using 1D array...
//it would be difficult to do the above since we want the operations to occur over abitrarily large matrices
//this can definitely be optimzied by elminating redundant calculations

//---------------------------------------------------------------------------------
__global__ void matrixDivideKernel(double* c, const double* a, const double* b) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < N) {
c[tid] = (a[tid]/b[tid]);
tid += blockDim.x * gridDim.x;
}
}