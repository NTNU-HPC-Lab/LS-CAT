#include "includes.h"
__global__ void kernel_mul(char* newB,  char* first, char* second, int size_first, int size_second, int * size_newB) {

int i = threadIdx.x;
int j = threadIdx.y;

int tid = j * gridDim.x * blockDim.x + i ;

if(j!=0 && i!=0){
newB[tid] = first[i] * second[j];
}

if(j==0 && i==0){
if(first[j] != second[i])
newB[0]='-';
else
newB[0]='+';
}
}