#include "includes.h"
__device__ int d(void) { return 8; }
__global__ void test_num_vgpr_num_sgpr() { }