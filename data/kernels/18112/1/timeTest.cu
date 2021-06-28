#include "includes.h"
__device__ void timeTest1(int *a){
int t_index = threadIdx.x + (blockIdx.x * blockDim.x);

if (t_index < SIZE) {
*a +=5;
}

}
__global__ void timeTest() {

int t_index = threadIdx.x + (blockIdx.x * blockDim.x);

if (t_index < SIZE) {

int a = 0;

for(int i = 0; i < 10000000; i++){
timeTest1(&a);
}

}
}