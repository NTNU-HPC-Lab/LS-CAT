#include "includes.h"
__device__ int test()
{
return 10;
}
__global__ void testDrive()
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
printf("%d\n", index);
//if (index == 0)
//{
int num = test();
printf("num = %d\n", num);
//}
}