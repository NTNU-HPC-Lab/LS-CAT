#include "includes.h"
__global__ void sortKernelMulti(int *arr, int arr_len, int num_elem, int oddEven)
{
int i = 2 * (blockIdx.x * blockDim.x * num_elem) + oddEven;
int iterEnd = min(arr_len - 1, i + 2 * blockDim.x *num_elem);
// Increment to thread start index:
i += 2 * threadIdx.x;
// Every thread in block (warp) step by num_elem
for (; i < iterEnd; i += 2 * blockDim.x)
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