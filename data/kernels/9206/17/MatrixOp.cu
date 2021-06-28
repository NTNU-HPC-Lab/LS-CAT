#include "includes.h"
__global__ void MatrixOp(int *arr, int N) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
int swapVar;
if(i<N && j<N) {
if(j%2==0 && (j+1)!=N) {
// swap elements
swapVar = arr[i*N + j];
arr[i*N + j] = arr[i*N+j+1];
arr[i*N+j+1] = swapVar;
}
__syncthreads();
if(i > j){
arr[j*N + i] = arr[i*N+j];
}
}
}