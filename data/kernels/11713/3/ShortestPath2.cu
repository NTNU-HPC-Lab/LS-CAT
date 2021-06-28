#include "includes.h"
__global__ void ShortestPath2(float *Arr1,float *Arr2,float *recv,int N,int rows, int k,int rank,int owner){

int col=blockIdx.x * blockDim.x + threadIdx.x;
int row=blockIdx.y * blockDim.y + threadIdx.y;

int index=row*N+col;
int index_ik = row*N+k;


if(Arr1[index]>(Arr1[index_ik]+recv[col])){
Arr2[index]=Arr1[index_ik]+recv[col];
}
__syncthreads();

}