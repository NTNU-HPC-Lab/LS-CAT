#include "includes.h"
__global__ void multiplication(int * A,int * B,int * C,int N,int M,int K){
int row = blockIdx.y*blockDim.y+threadIdx.y;
int col = blockIdx.x*blockDim.x+threadIdx.x;

if(row<N && col<K){//Si no me fui del arreglo
int sum=0;
for(int i=0;i<M;i++){
sum+=A[row*N+i]*B[i*M+col];
}
C[row*N+col]=sum;
}
}