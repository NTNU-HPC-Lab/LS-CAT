#include "includes.h"

__device__ int position;			//index of the largest value
__device__ int largest;				//value of the largest value
int lenString = 593;
int maxNumStrings = 1000000;
int threshold = 2;

__global__ void anyLeft(int *d_c, int *remaining, int size) {
int my_id = blockDim.x * blockIdx.x + threadIdx.x;
if((d_c[my_id] == 0) && (my_id < size)) {
*remaining = 0;
}
}