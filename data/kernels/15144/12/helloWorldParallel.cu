#include "includes.h"
__global__ void helloWorldParallel( void ) {
int i = threadIdx.x;
int j = blockIdx.x;
printf("Hello world from GPU %d/%d\n", j, i);
}