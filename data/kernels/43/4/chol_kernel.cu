#include "includes.h"
__global__ void chol_kernel(float * U, int ops_per_thread) {
//Determine the boundaries for this thread
//Get a thread identifier
int tx = blockIdx.x * blockDim.x + threadIdx.x;

//Iterators
unsigned int i, j, k;
//unsigned int size = MATRIX_SIZE*MATRIX_SIZE;
unsigned int num_rows = MATRIX_SIZE;

//Contents of the A matrix should already be in U

//Perform the Cholesky decomposition in place on the U matrix
for (k = 0; k < num_rows; k++) {
//Only one thread does squre root and division
if (tx == 0) {
// Take the square root of the diagonal element
U[k * num_rows + k] = sqrt(U[k * num_rows + k]);
//Don't bother doing check...live life on the edge!

// Division step
for (j = (k + 1); j < num_rows; j++) {
U[k * num_rows + j] /= U[k * num_rows + k]; // Division step
}
}

//Sync threads!!!!! (only one thread block so, ok)
__syncthreads();

//Elimination step

int istart = ( k + 1 )  +  tx * ops_per_thread;
int iend = istart + ops_per_thread;

for (i = istart; i < iend; i++) {
//Do work  for this i iteration
for (j = i; j < num_rows; j++) {
U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
}
}


//Sync threads!!!!! (only one thread block so, ok)
__syncthreads();
}

//Sync threads!!!!! (only one thread block so, ok)
__syncthreads();




//As the final step, zero out the lower triangular portion of U
//for(i = 0; i < U.num_rows; i++)
//Each thread does so many iterations of zero out loop
//Starting index for this thread
int istart = tx * ops_per_thread;
//Ending index for this thread
int iend = istart + ops_per_thread;

//Check boundaries, else do nothing
for (i = istart; i < iend; i++) {
//Do work  for this i iteration
for (j = 0; j < i; j++) {
U[i * num_rows + j] = 0.0;
}
}


//Don't sync, will sync outside here
}