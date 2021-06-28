#include "includes.h"

__device__ int position;			//index of the largest value
__device__ int largest;				//value of the largest value
int lenString = 593;
int maxNumStrings = 1000000;
int threshold = 2;

__global__ void search(int *d_b, int *d_c, int size) {
int my_id = blockDim.x * blockIdx.x + threadIdx.x;
if((d_c[my_id] == 0) && (d_b[my_id] == largest) && (my_id < size)) {
position = my_id;
}
}