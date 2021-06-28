#include "includes.h"
__global__ void matrixColour (float *a, float *b, int n){
int j= blockDim.x * blockIdx.x + threadIdx.x;

if(j<n){
for (int i=0; i<n; i++){
printf("Thread = %d ; i = %d ; %f\n",j+1,i+1,b[i]);
if (a[j*n+i]==1){
if (b[j]==b[i]){
b[j]=-1;
break;
}
}
}
}

//	int colour[10];
//
//	memset(colour, 0, 10*sizeof(float));

//	if (j<n){
//		for (int i=0; i<n; i++){
//			//printf("Thread = %d ; i = %d ; %f\n",j+1,i+1,b[i]);
//			if (a[j*n+i]==1 && b[i]!=-1){
//				colour[(int)b[i]]=1;
//			}
//
//
////			if (i==j){
////				//atomicAdd(&b[i],1.0f);
////				b[i]+=1.0f;
////			}
//		}
//
//		for (int i=0; i<n; i++){
//			if (colour[i]==0){
//				b[j]=i;
//				break;
//			}
//
//
//
//		}
//
//
//		for (int i=0; i<n; i++){
//			printf("Thread = %d ; i = %d ; %f\n",j+1,i+1,b[i]);
//
//
//		}
//
//	}




//	printf("I am thread no: %d from blocknumber: %d\n", threadIdx.x, blockIdx.x);

//b[j] = j+1;


}