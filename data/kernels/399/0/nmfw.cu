#include "includes.h"
using namespace std;

#define BLOCKSIZE 32

//test code
__global__ void nmfw(float *a, int r, int c, int k, float *w, float *h, float *wcp)//must be block synchronized!!!
{
int row = blockIdx.y*blockDim.y + threadIdx.y;
int col = blockIdx.x*blockDim.x + threadIdx.x;

//compute W
if (col < k && row < r) {
//ah'
float sum = 0.0;
float temp = 0.0;
for (int i = 0; i < c; i++)
sum += a[row*c + i]*h[col*c + i];
temp =  w[row*k+col]*sum;
//whh'
sum = 0.0;
for (int i = 0; i < c; i++) {
for (int j = 0; j < k; j++) {
sum += w[row*k + j]*h[j*c + i]*h[col*c+i];
}
}
__syncthreads();
wcp[row*k+col] = temp/sum;
}
}