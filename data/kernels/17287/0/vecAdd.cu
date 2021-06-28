#include "includes.h"
using namespace std;

int *a, *b;  // host data
int *c, *c2;  // results


__global__ void vecAdd(int *A,int *B,int *C,int N)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
C[i] = A[i] + B[i];
}