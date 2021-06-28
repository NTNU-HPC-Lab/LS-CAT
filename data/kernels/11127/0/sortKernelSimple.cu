#include "includes.h"
__global__ void sortKernelSimple(int *arr, int arr_len, int odd)
{
int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + odd;
if (i < arr_len - 1)
{
//Even
int a = arr[i];
int b = arr[i + 1];
if (a > b)
{
arr[i] = b;
arr[i + 1] = a;
}
}
}