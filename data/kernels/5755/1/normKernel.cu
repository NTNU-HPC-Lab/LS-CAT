#include "includes.h"

#define NUMBER_OF_BLOCKS 256
#define NUMBER_OF_THREADS 64

// ==========
// Macro taken from:
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
__device__ int getElementsPerUnit(int total, int number_of_units) {
int elements_per_unit = total / number_of_units;
double remains = total % number_of_units;

if(remains != 0) {
elements_per_unit += 1;
}

return elements_per_unit;
}
__global__ void normKernel(double *vectors, int size, double *results, int vector_size) {
int vectors_per_block = getElementsPerUnit(size, gridDim.x);

// Get range of vectors we will work with
int start = blockIdx.x * vectors_per_block;
int end = start + vectors_per_block;

if(end > size) {
end = size;
}

for(int vec_index = start; vec_index < end; vec_index++) {
for(int i = 0; i < vector_size; i++) {
results[vec_index] += pow(vectors[vec_index*vector_size + i], 2);
}

results[vec_index] = sqrt(results[vec_index]);
}
}