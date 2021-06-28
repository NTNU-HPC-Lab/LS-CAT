#include "includes.h"
/*Title: Vector addition and subtraction in CUDA.
A simple way to understand how CUDA can be used to perform arithmetic operations.
*/
using namespace std;
# define size 5

//Global functions

//********************************************************
__global__ void AddIntsCUDA(int *a, int *b)
{
int tid=blockIdx.x*blockDim.x+threadIdx.x;
a[tid] = a[tid] + b[tid];
}