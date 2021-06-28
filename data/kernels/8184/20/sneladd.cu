#include "includes.h"
__global__ void sneladd(float * inA, float * inB, int *sub, int Nprj, int snno)
{
int idz = threadIdx.x + blockDim.x*blockIdx.x;
if (blockIdx.y<Nprj && idz<snno)
inA[snno*blockIdx.y + idz] += inB[snno*sub[blockIdx.y] + idz];//sub[blockIdx.y]
}