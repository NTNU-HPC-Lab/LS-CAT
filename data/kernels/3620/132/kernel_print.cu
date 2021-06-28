#include "includes.h"
__global__ void kernel_print( int const* p, int n)
{
printf("int: %d ",n);
for(int i=0; i<n; i++)
printf("%d ",*(p+i));
}