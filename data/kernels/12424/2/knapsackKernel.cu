#include "includes.h"
__global__ void knapsackKernel(int *profits, int *weights, int *input_f, int *output_f, int capacity, int c_min, int k){

int c = blockIdx.x*512 + threadIdx.x;
if(c<c_min || c>capacity){return;}
if(input_f[c] < input_f[c-weights[k-1]]+profits[k-1]){
output_f[c] = input_f[c-weights[k-1]]+profits[k-1];
}
else{
output_f[c] = input_f[c];
}
}