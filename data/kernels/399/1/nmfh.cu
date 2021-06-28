#include "includes.h"
using namespace std;

#define BLOCKSIZE 32

//test code
__global__ void nmfh(float *a, int r, int c, int k, float *w, float *h, float *hcp)//must be block synchronized!!!
{
int row = blockIdx.y*blockDim.y + threadIdx.y;
int col = blockIdx.x*blockDim.x + threadIdx.x;

//compute H
if (row < k && col < c) {
//w'a
float temp = 0.0;
float sum;
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