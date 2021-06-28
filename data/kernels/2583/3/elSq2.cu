#include "includes.h"

extern "C"
{










}
__global__ void elSq2(int N, int M, float *In, float *Out)
{

int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int index = j*N + i;

if (i < N && j < M)
{
Out[index] = __fmul_rn(In[index], In[index]);
}
}