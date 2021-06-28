#include "includes.h"
__global__ void kernel_array_beam_slave_sincos_original(int N, float r1, float r2, float r3, float *x, float *y, float *z, float *sum, int blockDim_2) {
unsigned int n=threadIdx.x; //+blockDim.x*blockIdx.x;
__shared__ float tmpsum[1000]; /* assumed to be size 2*Nx1 */
if (n<N) {
float ss,cc;
sincosf((r1*__ldg(&x[n])+r2*__ldg(&y[n])+r3*__ldg(&z[n])),&ss,&cc);
tmpsum[2*n]=ss;
tmpsum[2*n+1]=cc;
}
__syncthreads();

// Build summation tree over elements, handling case where total threads is not a power of two.
int nTotalThreads = blockDim_2; // Total number of threads (==N), rounded up to the next power of two
while(nTotalThreads > 1) {
int halfPoint = (nTotalThreads >> 1); // divide by two
if (n < halfPoint) {
int thread2 = n + halfPoint;
if (thread2 < blockDim.x) { // Skipping the fictitious threads >N ( blockDim.x ... blockDim_2-1 )
tmpsum[2*n] = tmpsum[2*n]+tmpsum[2*thread2];
tmpsum[2*n+1] = tmpsum[2*n+1]+tmpsum[2*thread2+1];
}
}
__syncthreads();
nTotalThreads = halfPoint; // Reducing the binary tree size by two
}

/* now thread 0 will add up results */
if (threadIdx.x==0) {
sum[0]=tmpsum[0];
sum[1]=tmpsum[1];
}
}