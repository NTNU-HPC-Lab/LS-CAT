#include "includes.h"
__global__ void set_valid_pos_flag(int32_t* pos_buff, const int32_t* count_buff, const int32_t entry_count) {
const int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
const int32_t step = blockDim.x * gridDim.x;
for (int32_t i = start; i < entry_count; i += step) {
if (count_buff[i]) {
pos_buff[i] = VALID_POS_FLAG;
}
}
}