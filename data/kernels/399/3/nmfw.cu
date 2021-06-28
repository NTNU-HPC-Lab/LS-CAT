#include "includes.h"
__global__ void nmfw(double *a, int r, int c, int k, double *w, double *h, double *wcp)//must be block synchronized!!!
{
int row = blockIdx.y*blockDim.y + threadIdx.y;
int col = blockIdx.x*blockDim.x + threadIdx.x;

//compute W
if (col < k && row < r) {
//ah'
double sum = 0.0;
double temp = 0.0;
for (int i = 0; i < c; i++)
sum += a[row*c + i]*h[col*c + i];
temp = w[row*k+col]*sum;
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