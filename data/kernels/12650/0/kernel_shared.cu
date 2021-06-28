#include "includes.h"


#define FIBER 32
#define MATRIX_SIZE 2048
#define DATA_SIZE MATRIX_SIZE * MATRIX_SIZE * sizeof(int)
#define MAX_MATRIX_SIZE (MATRIX_SIZE * MATRIX_SIZE)



using namespace std;

__global__ void kernel_shared(int *A, int *C, int *B, int *result) {
__shared__ int shared_memory[FIBER][FIBER];

int i = blockIdx.x * blockDim.x + threadIdx.y;
int j = blockIdx.y * blockDim.y + threadIdx.x;

shared_memory[threadIdx.y][threadIdx.x] = B[i * MATRIX_SIZE + j];

__syncthreads();

i = blockIdx.x * blockDim.x + threadIdx.x;
j = blockIdx.y * blockDim.y + threadIdx.y;

int first_index = i + j * MATRIX_SIZE;
int second_index = j + i * MATRIX_SIZE;

if (first_index < MAX_MATRIX_SIZE && second_index < MAX_MATRIX_SIZE)
{
result[first_index] = (A[first_index] + A[first_index]) * shared_memory[threadIdx.x][threadIdx.y] - C[first_index];
}
}