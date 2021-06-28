#include "includes.h"
__global__ void unique_index_kernel(const char* flag, const int* flag_inc_sum, int* unique_index, int num_elems) {
int gid_base = blockIdx.x * blockDim.x + threadIdx.x;
for (int gid = gid_base; gid < num_elems; gid += blockDim.x * gridDim.x) {
if (flag[gid] == 1) {
int id = flag_inc_sum[gid] - 1;
unique_index[id] = gid;
}
}
}