#include "includes.h"


#define DATA_SIZE (1024 * 1024 * 256)
#define DATA_RANGE (256)

void printHist(int * arr, char * str);




__global__ void histogram_shared(float * a, int * histo, int n) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
__shared__ int sh[DATA_RANGE];

if(threadIdx.x < 256) sh[threadIdx.x] = 0;
__syncthreads();

if(tid < n) atomicAdd(&sh[(int)a[tid]], 1);
__syncthreads();

if(threadIdx.x < 256) atomicAdd(&histo[threadIdx.x], sh[threadIdx.x]);

}