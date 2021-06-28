#include "includes.h"
void vectorAdd(double* A, double* B,double* C,int n);

__global__ void vecAddKernel(double* A, double* B, double* C, int n)	{
int i=blockDim.x*blockIdx.x+threadIdx.x;
if(i<n) {
C[i]=A[i]+B[i];
}
}