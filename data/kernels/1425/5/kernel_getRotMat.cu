#include "includes.h"
__global__ void kernel_getRotMat(double* devRotm, double* devnR, int nR)
{
extern __shared__ double matS[];
int tid = threadIdx.x + blockIdx.x * blockDim.x;

if (tid >= nR)
return;

double *mat, *res;
mat = matS + threadIdx.x * 18;
res = mat  + 9;

mat[0] = 0; mat[4] = 0; mat[8] = 0;
mat[5] = devnR[tid * 4 + 1];
mat[6] = devnR[tid * 4 + 2];
mat[1] = devnR[tid * 4 + 3];
mat[7] = -mat[5];
mat[2] = -mat[6];
mat[3] = -mat[1];

for(int i = 0; i < 9; i++)
res[i] = 0;

for (int i = 0; i < 3; i++)
for (int j = 0; j < 3; j++)
for (int k = 0; k < 3; k++)
res[i + j * 3] += mat[i + k * 3] * mat[k + j * 3];

double scale = 2 * devnR[tid * 4];
for (int n = 0; n < 9; n++)
{
mat[n] *= scale;
mat[n] += res[n] * 2;
}

mat[0] += 1;
mat[4] += 1;
mat[8] += 1;

for (int n = 0; n < 9; n++)
{
devRotm[tid * 9 + n] = mat[n];
}
}