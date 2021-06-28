#include "includes.h"

#define NUMBER_OF_BLOCKS 256
#define NUMBER_OF_THREADS 64

// ==========
// Macro taken from:
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
__device__ double dotProduct(double *a, double *b, int size) {
double result = 0;

for(int i = 0; i < size; i++) {
result += a[i] * b[i];
}

return result;
}
__device__ int getElementsPerUnit(int total, int number_of_units) {
int elements_per_unit = total / number_of_units;
double remains = total % number_of_units;

if(remains != 0) {
elements_per_unit += 1;
}

return elements_per_unit;
}
__global__ void dotProductKernel(double *a, double *b, int a_size, int b_size, double *results, int vector_size) {
int a_vectors_per_block = getElementsPerUnit(a_size, gridDim.x);
int b_vectors_per_thread = getElementsPerUnit(b_size, blockDim.x);

// Get range of 'a' vectors we will work with
int a_start = blockIdx.x * a_vectors_per_block;
int a_end = a_start + a_vectors_per_block;

if(a_end > a_size) {
a_end = a_size;
}

// Get range of 'b' vectors we will work with
int b_start = threadIdx.x * b_vectors_per_thread;
int b_end = b_start + b_vectors_per_thread;

if(b_end > b_size) {
b_end = b_size;
}

if(a_start < a_size && b_start < b_size) {
for(int a_index = a_start; a_index < a_end; a_index++) {
for(int b_index = b_start; b_index < b_end; b_index++) {
results[a_index*b_size + b_index] = dotProduct(&a[a_index*vector_size], &b[b_index*vector_size], vector_size);
}
}
}
}