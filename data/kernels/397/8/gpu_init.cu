#include "includes.h"
__global__ void gpu_init(int *mapad, int max, int size)
{
/*Identificaciones necesarios*/
/*int IDX_Thread = threadIdx.x;
int IDY_Thread = threadIdx.y;
int IDX_block =	blockIdx.x;
int IDY_block =	blockIdx.y;
int shapeGrid_X = gridDim.x;
int threads_per_block =	blockDim.x * blockDim.y;
int position = threads_per_block * ((IDY_block * shapeGrid_X)+IDX_block)+((IDY_Thread*blockDim.x)+IDX_Thread);*/

int position = blockDim.x * blockDim.y * ((blockIdx.y * gridDim.x)+blockIdx.x)+((threadIdx.y*blockDim.x)+threadIdx.x);

if (position<size) mapad[position] = max;
}