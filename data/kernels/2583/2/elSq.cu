#include "includes.h"

extern "C"
{










}
__global__ void elSq(int N, int M, float *Mat)
{

int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int index = j*N + i;

if (i < N && j < M)
{
Mat[index] = __fmul_rn(Mat[index], Mat[index]);
}
}