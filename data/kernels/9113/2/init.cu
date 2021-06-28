#include "includes.h"
__global__ void init(int *arr, int sqroot, int limit) {
int c;
for(c = 2; c <= sqroot; c++) {
if(arr[c] == 0) {
/*
#pragma omp parallel for shared(arr, limit, c) private(m)
for(m = c+1; m < limit; m++) {
if(m%c == 0) {
arr[m] = 1;
}
}
*/
int tid = c+1+ threadIdx.x + (blockIdx.x * blockDim.x);
if (tid<limit){
if (tid % c ==0) {
arr[tid] = 1;
}
}


}
}
}