#include "includes.h"

__device__ int position;			//index of the largest value
__device__ int largest;				//value of the largest value
int lenString = 593;
int maxNumStrings = 1000000;
int threshold = 2;

__global__ void compare(char *d_a, int *d_b, int *d_c, int size, int lenString, int threshold) {

int my_id = blockDim.x * blockIdx.x + threadIdx.x;

if (my_id == position)
d_c[my_id] = 2;


if ((my_id < size) && (d_c[my_id] == 0) && (my_id != position)) {
int x, diffs = 0;

for (x = 0; x < lenString; x++) {
diffs += (bool)(d_a[(lenString*position)+x]^d_a[(my_id*lenString)+x]);

if (diffs > threshold)
break;
}

if (diffs <= threshold) {
d_b[position] += d_b[my_id];
d_c[my_id] = 1;
}
}
}