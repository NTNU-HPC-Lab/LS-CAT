#include "includes.h"
__global__ void kernel_2_shared(int columns, int rows, float* mat1, float* matanswer) {
__shared__ float temp_answer[32];
if (threadIdx.y == 0) {
temp_answer[threadIdx.x] = float(0);
}
__syncthreads();  //Existe pero no es reconocido por Itellisense

int thread_mat_colid = blockIdx.x * blockDim.x + threadIdx.x;
int thread_mat_rowid = blockIdx.y * blockDim.y + threadIdx.y;
int position_in_matrix = thread_mat_rowid * columns + thread_mat_colid;

atomicAdd(&(temp_answer[threadIdx.x]), mat1[position_in_matrix]);
__syncthreads();  //Existe pero no es reconocido por Itellisense

if (blockIdx.x == 1 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
printf("%d Value %f \n", position_in_matrix, mat1[position_in_matrix]);
}

if (threadIdx.y == rows - 1) {
atomicAdd(&(matanswer[thread_mat_colid]), temp_answer[threadIdx.x]);
}
__syncthreads();
//printf("BlockID_x: %d BlockID_y: %d Blockdim_x: %d  Blockdim_y: %d ThreadIdx_x: %d ThreadIdx_y: %d \n",  blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);
}