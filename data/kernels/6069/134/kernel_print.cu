#include "includes.h"
__global__ void kernel_print( size_t const* p, int n)
{
printf("ulong: %d ",n);
for(int i=0; i<n; i++)
printf("%lu ",*(p+i));
}