#include "includes.h"

__device__ int add_d(int a, int b)
{
printf("Hellow world_3\n");
return a * b;
}
__global__ void add2(int a, int b, int *c)
{
printf("Hellow world_2\n");
*c = add_d(a, b);
printf("Hellow world_4\n");
}