#include "includes.h"
__global__ void nmfh(double *a, int r, int c, int k, double *w, double *h, double *hcp)//must be block synchronized!!!
{
int row = blockIdx.y*blockDim.y + threadIdx.y;
int col = blockIdx.x*blockDim.x + threadIdx.x;

//compute H
if (row < k && col < c) {
//w'a
double temp = 0.0;
double sum;
sum = 0.0;
for (int i = 0; i < r; i++)
sum += w[i*k + row]*a[i*c+col];

temp = h[row*c+col]*sum;
//w'wh
sum = 0.0;
for (int i = 0; i < k; i++)
for (int j = 0; j < r; j++)
sum += w[j*k + row]*w[j*k + i]*h[i*c+col];

__syncthreads();
hcp[row*c+col] = temp/sum;
}
}