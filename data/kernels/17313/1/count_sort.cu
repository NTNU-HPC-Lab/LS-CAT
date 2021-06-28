#include "includes.h"
__global__ void count_sort(int *x, int *y, int size){

int idx = blockIdx.x * blockDim.x + threadIdx.x;

if(idx < size){

int count = 0;

for(int j = 0; j < size; j++){

if (x[j] < x[idx])
count++;
else if (x[j] == x[idx] && j < idx)
count++;

}

y[count] = x[idx];
}

}