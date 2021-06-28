#include "includes.h"
__global__ void loop()
{
/*
* This kernel does the work of only 1 iteration
* of the original for loop. Indication of which
* "iteration" is being executed by this kernel is
* still available via `threadIdx.x`.
*/

printf("This is iteration number %d\n", threadIdx.x);
}