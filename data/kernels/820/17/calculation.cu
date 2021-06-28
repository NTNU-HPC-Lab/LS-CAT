#include "includes.h"
__global__ void calculation(    char* dev_a, char* dev_b, char* dev_c, int num_matrices, int matrix_size ) {
// Each thread handles a matrix
int k = (blockIdx.x*blockDim.x) + threadIdx.x;    // this thread handles the data at its thread id

if (k >= num_matrices) return;

// If first element is different than 0 do the computation
if (dev_a[k*matrix_size*matrix_size] != 0){
for (int j = 0; j < matrix_size; j++){
//If first value in the row of the matrix, do addition
if (dev_a[k*matrix_size*matrix_size+j*matrix_size] < threshold){
for (int i = 0; i < matrix_size; i++){
int index = k*matrix_size*matrix_size+j*matrix_size+i;
dev_c[index] = dev_a[index] + dev_b[index];
}
//Do subtraction
} else {
for (int i = 0; i < matrix_size; i++){
int index = k*matrix_size*matrix_size+j*matrix_size+i;
dev_c[index] = dev_a[index] - dev_b[index];
}
}
}
}
}