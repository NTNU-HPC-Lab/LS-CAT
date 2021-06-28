#include "includes.h"
__global__ void decrementalColouringNew (int *vertexArray, int *neighbourArray, int n, int m, int *decrementalArray, int size){

int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i >= size){
return;
}



int startStart, startStop;
int me, you;
//	int otheri;
//	bool ipercent2 = false;

me = decrementalArray[i];

if (i%2 == 0){
you = decrementalArray[i+1];
//		otheri = i+1;
//		ipercent2 = true;
}
else{
you = decrementalArray[i-1];
//		otheri = i-1;
}

//printf("I am %d and I am deleting %d - %d\n", i, me, you);

startStart = vertexArray[me-1];

startStop = vertexArray[me];

for (int j=startStart; j<startStop; j++){
if (neighbourArray[j]==you){
neighbourArray[j]=0;
break;
}
}
}