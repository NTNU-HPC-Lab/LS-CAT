#include "includes.h"
__global__ void mk_kernel(char* keep_mem, size_t bytes)
{
for (unsigned i=0; i<bytes; ++i)
{
keep_mem[i] = 0;
}
}