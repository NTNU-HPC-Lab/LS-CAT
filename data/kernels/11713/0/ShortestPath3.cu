#include "includes.h"
#define BLOCKSIZE 4
#define CELLS_PER_THREAD 4     // Stride length
__global__ void ShortestPath3(float *Arr1,float *Arr2,int N){//Arr1 input array,Holds weights
//Arr2 output array
unsigned int k;
int row=blockIdx.x;
int col=threadIdx.x;

if(row >= N || col >= N) return;

int index=row*N+col;					//Index for Elements of global memory array
extern __shared__ float sArr[];			//Dynamic share memory allocation in Array
int stride=N/CELLS_PER_THREAD;			// Stride for each index of arrays (both global and shared)

for(stride=0;stride<N;stride+=N/CELLS_PER_THREAD){
if(threadIdx.x+stride>N) break;						//Copy elements of same ROW in shared memory (SHARED memory indexes = ThreadIdx.x + stride)
sArr[threadIdx.x+stride]=Arr1[index+stride];		//"jump" to the next index according to stride .Write to SM
}
__syncthreads();										//Wait all threads in current block


for(stride=0;stride<N;stride+=N/CELLS_PER_THREAD){

for(k=0;k<N;k++){
if(k==threadIdx.x+stride)continue;
if(sArr[threadIdx.x+stride]>(sArr[k]+Arr1[k*N+col+stride])){		//Return
Arr2[index+stride]=(sArr[k]+Arr1[k*N+col+stride]);		//If Condition true,write to shared memory
//
}
}
//						//Copy results to global memory and return to HOST

}
}