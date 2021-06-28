#include "includes.h"
// Template for Assignment 1: CUDA
// Use "icc -O -openmp" to compile

#define threshold 1e-4
#define n (2048)
void init(void);
void ref(void);
void test(void);
void compare(int N, double *wref, double *w);


__global__ void test_kernel(int N, double *A, double *B, double *X)
{
int i,j,k;
double temp;
// Template version uses only one thread, which does all the work
// This must be changed (and the launch parameters) to exploit GPU parallelism
// You can make any changes; only requirement is that correctness test passes
k = (blockIdx.y*gridDim.x+blockIdx.x)*(blockDim.x*blockDim.y)+(threadIdx.y*blockDim.x+threadIdx.x);
//if(threadIdx.x == 0) {
//for(k=0;k<n;k++){
/*
if(k<n){
for (i=0;i<n;i++){
temp = B[k*N+i]; // temp = b[k][i];
for (j=0;j<i;j++) temp = temp - A[i*N+j] * X[k*N+j]; // temp = temp - a[i][j]*x[k][j];
X[k*N+i] = temp/A[i*N+i]; //x[k][i] = temp/a[i][i];
}
}
*/
if(k<n){
for (i=0;i<n;i++){
temp = B[i*N+k]; // temp = b[k][i];
for (j=0;j<i;j++) temp = temp - A[j*N+i] * X[j*N+k]; // temp = temp - a[i][j]*x[k][j];
X[i*N+k] = temp/A[i*N+i]; //x[k][i] = temp/a[i][i];
}
}
//  }
// }
}