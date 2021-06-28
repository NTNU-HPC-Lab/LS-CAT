#include "includes.h"

extern "C"
{










}
__global__ void scaleParams(int N, int M, float c, float *Mat, float *F)
{

int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int index = j*N + i;

if (i < N && j < M)
{
float s = __saturatef( __fdividef(c, __fsqrt_rn(F[i])));
//float s = (c/sqrt(F[i]) < 1.0) ? c/sqrt(F[i]) : 1.0;
Mat[index] = __fmul_rn(Mat[index], s);
}
}