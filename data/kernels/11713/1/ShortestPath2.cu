#include "includes.h"
#define BLOCKSIZE 4
#define CELLS_PER_THREAD 4     // Stride length
__global__ void ShortestPath2(float *Arr1,float *Arr2,int N){   //Arr1 input array,Holds weights
//Arr2 output array
unsigned int k;

int row=blockIdx.x;
int col=threadIdx.x;
if(row >= N || col >= N) return;

int index=row*N+col;						//Index for Elements of global memory array
extern __shared__ float sArr[];				//Dynamic share memory allocation in Array
Arr2[index]=Arr1[index];
sArr[threadIdx.x]=Arr1[index];				//Copy elements of same ROW in shared memory (SHARED memory indexes = ThreadIdx.x )
__syncthreads();							//Wait threads in block to finish

for(k=0;k<N;k++){
if(k==threadIdx.x) continue;						//If out of bounds , just current loop iteration
if(sArr[threadIdx.x]>(sArr[k]+Arr1[k*N+col])){		//If condition true write in shared memory new value
Arr2[index]=(sArr[k]+Arr1[k*N+col]);
}

}
}