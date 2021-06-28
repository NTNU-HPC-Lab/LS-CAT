#include "includes.h"
__device__ void generate2DGaussian(double * output, double sigma, int sz, bool normalize) {

/*x and y coordinates of thread in kernel. The gaussian filters are
*small enough for the kernel to fit into a single thread block of sz*sz*/
const int colIdx = threadIdx.x;
const int rowIdx = threadIdx.y;
int linearIdx = rowIdx*sz + colIdx;

/*calculate distance from centre of filter*/
int distx = abs(colIdx - sz/2);
int disty = abs(rowIdx - sz/2);

output[linearIdx] = exp(-(pow((double)(distx), 2.0)+pow((double)(disty), 2.0))/(2*(pow(sigma, 2.0))));

if(normalize==true) {

/*wait until all threads have assigned a value to their index in the output array*/
__syncthreads();

int i, j;
double sum=0.0;

for(i=0; i<sz; i++) {
for(j=0; j<sz; j++) {
sum += output[i*sz + j];
}
}

/*Let all threads calculate the sum before changing the value of the output array*/
__syncthreads();

output[linearIdx]/=sum;
}
}
__global__ void getDoG(double * output, double sigma, double sigmaratio) {

int sz = ceil(sigma*3) * 2 + 1;
int linearIdx = threadIdx.y*sz + threadIdx.x;
if(linearIdx>=sz*sz) return;

__shared__ double g1[900];
__shared__ double g2[900];

generate2DGaussian(g1, sigma, sz, true);
generate2DGaussian(g2, sigma*sigmaratio, sz, true);

__syncthreads();

output[linearIdx] = g2[linearIdx]-g1[linearIdx];
}