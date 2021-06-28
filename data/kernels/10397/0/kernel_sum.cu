#include "includes.h"

static const int n_el = 512;
static const size_t size = n_el * sizeof(float);
// declare the kernel function


// function which invokes the kernel
__global__ void kernel_sum(const float* A, const float* B, float* C, int n_el)
{
// calculate the unique thread index
int tid = blockDim.x * blockIdx.x + threadIdx.x;
// perform tid-th elements addition
if (tid < n_el) C[tid] = A[tid] + B[tid];
}