#include "includes.h"
#define BLOCKSIZE 4
#define CELLS_PER_THREAD 4     // Stride length
__global__ void ShortestPath1(float *Arr1,float *Arr2,int N){
//Arr1 input array,Holds  of (u,v)
//Arr2 output array
int k;
int col=blockIdx.x * blockDim.x + threadIdx.x;
int row=blockIdx.y * blockDim.y + threadIdx.y;
int index=row*N+col;

if((row<N)&&(col<N)){
Arr2[index]=Arr1[index];

for(k=0;k<N;k++){

if(Arr1[index]>(Arr1[row*N+k]+Arr1[N*k+col])){
Arr2[index]=Arr1[row*N+k]+Arr1[N*k+col];
//	printf("ENTERED %f \n",Arr2[index]);
}
}
}

}