#include "includes.h"


#define DATA_SIZE (1024 * 1024 * 256)
#define DATA_RANGE (256)

void printHist(int * arr, char * str);




__global__ void histogram_atomic(float * a, int * histo, int n) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if(tid >= n) return;
atomicAdd(histo + (int)a[tid], 1);
}