#include "includes.h"
__global__ void matmul_v1(float* a,float* b,float* c, int n){
// C(nxn) = A(nxn) * B(nxn);

__shared__ float A[TILE_SIZE][TILE_SIZE+1];
__shared__ float B[TILE_SIZE][TILE_SIZE+1];

int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;

int i = bx*TILE_SIZE+tx;
int j = by*TILE_SIZE+ty;

A[ty][tx] = A[ty][tx] = 0;
if(i >= n || j >= n) return;

float c_ij = 0;
for(int m=0;m<float(n)/TILE_SIZE;m++){
A[ty][tx] = a[j*n+ m*TILE_SIZE + tx];
B[ty][tx] = b[(m*TILE_SIZE+ty)*n+i];

//		printf("%d %d : %f - %f\n",tx,ty,A[ty][tx],B[ty][tx]);

__syncthreads();

for(int k=0;k<TILE_SIZE;k++)
c_ij += A[ty][k]*B[k][tx];
__syncthreads();
}
c[n*j+i] = c_ij;

}