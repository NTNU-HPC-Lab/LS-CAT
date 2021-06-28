#include "includes.h"
__global__ void row_sums(const float *A, float *sums, size_t ds){

int idx = threadIdx.x+blockDim.x*blockIdx.x; // create typical 1D thread index from built-in variables
if (idx < ds){
float sum = 0.0f;
for (size_t i = 0; i < ds; i++)
sum += A[idx*ds+i];         // write a for loop that will cause the thread to iterate across a row, keeeping a running sum, and write the result to sums
sums[idx] = sum;
}}