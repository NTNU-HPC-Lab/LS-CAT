#include "includes.h"
__global__ void init_group(int8_t* groups, const size_t group_count, const size_t col_count, const size_t* col_widths, const size_t* init_vals) {
const auto thread_index =
threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
if (thread_index >= group_count) {
return;
}
for (size_t i = 0; i < col_count; groups += col_widths[i++] * group_count) {
switch (col_widths[i]) {
case 4:
*reinterpret_cast<uint32_t*>(groups) =
*reinterpret_cast<const uint32_t*>(init_vals + i);
break;
case 8:
reinterpret_cast<size_t*>(groups)[thread_index] = init_vals[i];
break;
default:;
}
}
}