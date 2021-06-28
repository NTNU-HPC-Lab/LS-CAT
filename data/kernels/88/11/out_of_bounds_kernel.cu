#include "includes.h"
__device__ void out_of_bounds_function(void) {
*(int*) 0x87654320 = 42;
}
__global__ void out_of_bounds_kernel(void) {
out_of_bounds_function();
}