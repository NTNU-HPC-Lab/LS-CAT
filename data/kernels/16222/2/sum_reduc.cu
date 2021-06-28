#include "includes.h"
__global__ void sum_reduc(int* data, int* len, int* width){
int indx = blockIdx.x * gridDim.x + threadIdx.x;
int sum = 0;
for (int i=indx; i<indx + *width; i++){
if (i < *len)
sum += data[i];
}
data[indx] = sum;
}