#include "includes.h"
__global__ void matrixColour (float *a, float *b, int n){
int j= blockDim.x * blockIdx.x + threadIdx.x;

printf("Block = %d ; Thread = %d \n",blockIdx.x+1, threadIdx.x+1);

//	if(j<n){
//		for (int i=0; i<n; i++){
////			printf("Block = %d ; Thread = %d ; i = %d ; %f\n",blockIdx.x+1, j+1,i+1,b[i]);
//			if (a[j*n+i]==1){
//				if (b[j]==b[i]){
//					b[j]=-1;
//					break;
//				}
//			}
//		}
//	}

int *colour = new int[n];

memset(colour, 0, n*sizeof(int));

if (j<n){
for (int i=0; i<n; i++){
//printf("Thread = %d ; i = %d ; %f\n",j+1,i+1,b[i]);
printf("Block = %d ; Thread = %d First For i = %d\n",blockIdx.x+1, threadIdx.x+1, i+1);
if (a[j*n+i]==1 && b[i]!=-1){
colour[(int)b[i]]=1;
}


//			if (i==j){
//				//atomicAdd(&b[i],1.0f);
//				b[i]+=1.0f;
//			}
}

for (int i=0; i<n; i++){

if (colour[i]==0){
printf("Block = %d ; Thread = %d Second For i = %d\n",blockIdx.x+1, threadIdx.x+1, i+1);
atomicAdd(&b[j],(float)i-b[j]);
break;
}
}


//		for (int i=0; i<n; i++){
//			printf("Third Block = %d ; ThreadId = %d ; Thread = %d ; i = %d ; %f\n",blockIdx.x+1, threadIdx.x+1, j+1,i+1,b[i]);
//		}

}




//	printf("I am thread no: %d from blocknumber: %d\n", threadIdx.x, blockIdx.x);

//b[j] = j+1;


}