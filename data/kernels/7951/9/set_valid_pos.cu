#include "includes.h"
__global__ void set_valid_pos(int32_t* pos_buff, int32_t* count_buff, const int32_t entry_count) {
const int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
const int32_t step = blockDim.x * gridDim.x;
for (int32_t i = start; i < entry_count; i += step) {
if (VALID_POS_FLAG == pos_buff[i]) {
pos_buff[i] = !i ? 0 : count_buff[i - 1];
}
}
}