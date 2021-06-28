#include "includes.h"

__device__ int position;			//index of the largest value
__device__ int largest;				//value of the largest value
int lenString = 593;
int maxNumStrings = 1000000;
int threshold = 2;

__global__ void populate (int *d_b, int *copy_db, int *d_c, int size, int *left) {
int n = 0;
*left = 1;	// reinitalized to false to check if all strings are merged

int my_id = blockDim.x * blockIdx.x + threadIdx.x;

if (my_id < size) {
n = abs((bool)d_c[my_id] - 1);
copy_db[my_id] = d_b[my_id] * n;
}
}