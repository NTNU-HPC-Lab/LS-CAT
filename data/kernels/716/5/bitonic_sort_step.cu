#include "includes.h"
__global__ void bitonic_sort_step(int *dev_values, int j, int k)
{
unsigned int i, ixj; /* Sorting partners: i and ixj */
i = threadIdx.x + blockDim.x * blockIdx.x;
ixj = i^j;

/* The threads with the lowest ids sort the array. */
if ((ixj)>i) {
if ((i&k)==0) {
/* Sort ascending */
if (dev_values[i]>dev_values[ixj]) {
// swap
int temp = dev_values[i];
dev_values[i] = dev_values[ixj];
dev_values[ixj] = temp;
}
}
if ((i&k)!=0) {
/* Sort descending */
if (dev_values[i]<dev_values[ixj]) {
// swap
int temp = dev_values[i];
dev_values[i] = dev_values[ixj];
dev_values[ixj] = temp;
}
}
}
}