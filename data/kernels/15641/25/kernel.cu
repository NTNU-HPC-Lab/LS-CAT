#include "includes.h"
__global__ void kernel(int* arr,int offset_min,int n){

int bx = blockIdx.x;
int tx = threadIdx.x;

int BX = blockDim.x;

int i = bx*BX+tx;

if (i>= n|| i < 0) return;
//printf("%d %d - %d %d\n",offset_min,offset_max,i+offset_min,i);
arr[i+offset_min] += 1;

}