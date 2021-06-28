#include "includes.h"
__global__ void kernel_print( long const* p, int n)
{
printf("long: %d ",n);
for(int i=0; i<n; i++)
printf("%ld ",*(p+i));
}