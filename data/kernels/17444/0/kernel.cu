#include "includes.h"

__global__ void kernel(float * w_vect, float * train, float * partition, int rows, int cols){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int i=0;
float temp = 0;
for(i = 0; i<cols; i++){
temp += w_vect[i]*train[i*rows+tid];
}
partition[tid] = temp;
}