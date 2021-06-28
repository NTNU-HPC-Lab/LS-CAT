#include "includes.h"
__global__ void matmul_v0(float* a,float* b,float* c, int n){
// C(nxn) = A(nxn) * B(nxn);
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;

if(i >= n || j >= n) return;

float c_ij = 0;
for(int k=0;k<n;k++){
c_ij += a[n*j+k]*b[n*k+i];

//		printf("%d %d %d : %f %f\n",i,j,k,a[n*j+k],b[n*k+i]);

}
c[n*j+i] = c_ij;

}