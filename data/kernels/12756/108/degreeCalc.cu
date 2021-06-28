#include "includes.h"
__device__ int sum = 1;  __global__ void degreeCalc (int *array){

int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i>=1000000){
return;
}

sum+=array[i];

//	if (i==999999){
//		printf("%d", sum);
//	}
}
__global__ void degreeCalc (int *vertexArray, int *neighbourArray, int *degreeCount, int n, int m){

int i= blockDim.x * blockIdx.x + threadIdx.x;

if (i>=n){
return;
}


int start = -1, stop = -1;
int diff=0;

start = vertexArray[i];

stop = vertexArray[i+1];


diff = stop-start;

degreeCount[i]=diff;
}