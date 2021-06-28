#include "includes.h"
__global__ void kernel1(int* D, int* q, int b){

int i = threadIdx.x + b * THR_PER_BL;
int j = threadIdx.y + b * THR_PER_BL;

float d, f, e;
for(int k = b * THR_PER_BL; k < (b + 1) * THR_PER_BL; k++)
{
d = D[i * N + j];
f = D[i * N + k];
e = D[k * N + j];

__syncthreads();

if(d > f + e)
{
D[i * N + j] = f + e;
q[i * N + j] = k;
}
}
}