#include "includes.h"
__global__ void spinKernel(unsigned long long timeout_clocks = 100000ULL)
{
register unsigned long long start_time, sample_time;
start_time = clock64();
while(1) {
sample_time = clock64();
if (timeout_clocks != ~0ULL && (sample_time - start_time) > timeout_clocks) {
break;
}
}
}