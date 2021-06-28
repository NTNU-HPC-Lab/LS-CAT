#include "includes.h"
__global__ void initArray(uint32_t * path, double *approx, uint32_t *top_k, int n){
int index = threadIdx.x + blockIdx.x * blockDim.x;
if(index < n){
for(int i = 0; i < sizeof(path); i++){
approx[i]++;
top_k[i] = path[i]++;
}
}
}