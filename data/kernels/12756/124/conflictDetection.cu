#include "includes.h"
__global__ void conflictDetection (int *vertexArray, int *neighbourArray, int *degreeCount, int n, int m, int *detectConflict){

int i= blockDim.x * blockIdx.x + threadIdx.x;

if (i>=n){
return;
}

int myColour = degreeCount[i];

int start = -1, stop = -1;

start = vertexArray[i];


stop = vertexArray[i+1];

for (int j=start; j<stop; j++){
if (degreeCount[neighbourArray[j]-1] == myColour){

//			detectConflict[i]=1;
//			break;

if (i < neighbourArray[j]-1){
if (detectConflict[i]!=1){
detectConflict[i]=1;
}
}
else if (detectConflict[neighbourArray[j]-1]!=1){
detectConflict[neighbourArray[j]-1]=1;
}






//			if (detectConflict[i]!=1){
//				detectConflict[i]=1;
//			}
//
//			if (detectConflict[neighbourArray[j]-1]!=1){
//				detectConflict[neighbourArray[j]-1]=1;
//			}
}
}
}