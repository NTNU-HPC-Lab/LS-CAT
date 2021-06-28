#include "includes.h"
__global__ void frontier_tail_swap_kernel(int* p_frontier_tail_d, int* c_frontier_tail_d) {

*p_frontier_tail_d = *c_frontier_tail_d;
*c_frontier_tail_d = 0;
}