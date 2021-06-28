#include "includes.h"

extern "C"
{










}
__global__ void updateEst(int N, int M, float beta2, float scale, float *PARAMS, float *AVG, float *EST)
{

int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int index = j*N + i;

float beta2a = __fsub_rn(1.0, beta2);
if (i < N && j < M)
{
//AVG[index] = beta2*AVG[index] + (1.0-beta2)*PARAMS[index];
//EST[index] = scale*AVG[index];
AVG[index] = __fmaf_rn(beta2a,PARAMS[index],__fmul_rn(beta2,AVG[index]));
EST[index] = __fmul_rn(scale, AVG[index]);
}
}