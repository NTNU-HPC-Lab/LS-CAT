#include "includes.h"
__global__ void ShortestPath1(float *Arr1,float *Arr2,int N,int rows, int rank){     //rowNum is number of rows for each process (full assigned to process)
//Arr1 input array,Holds  of (u,v)	//Arr2 output array
int k;
int col=blockIdx.x * blockDim.x + threadIdx.x;
int row=blockIdx.y * blockDim.y + threadIdx.y;

int offset=rows*rank;

int index=row*N+col;
int index_ik,index_kj;

Arr2[index]=Arr1[index];
for(k=rank*rows; k<((rank+1)*rows); k++){

index_ik = row*N+k;
index_kj = (k-offset)*N+col;

if(Arr1[index]>(Arr1[index_ik]+Arr1[index_kj])){
Arr2[index]=Arr1[index_ik]+Arr1[index_kj];
}
__syncthreads();
}
}