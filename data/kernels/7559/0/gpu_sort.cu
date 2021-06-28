#include "includes.h"

#define THREADS_PER_BLOCK 256
#define CHUNK_SIZE 16

// flag if the prng has been seeded
int randNotSeeded = 1;

// tests the gpu merge sort
__device__ void insertionSort(int *array, int a, int b)
{
int current;
for (int i = a + 1; i < b; i++)
{
current = array[i];
for (int j = i - 1; j >= a - 1; j--)
{
if (j == a - 1 || current > array[j])
{
array[j + 1] = current;
break;
}
else
{
array[j + 1] = array[j];
}
}
}
}
__global__ void gpu_sort(int *d_array, int size, int chunkSize)
{
// Figure out left and right for this thread
int a = (threadIdx.x + blockDim.x * blockIdx.x) * chunkSize;
if (a >= size) return;

int b = a + chunkSize;
if (b > size) b = size;

insertionSort(d_array, a, b);
}