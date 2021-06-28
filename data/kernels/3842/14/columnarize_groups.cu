#include "includes.h"
__global__ void columnarize_groups(int8_t* columnar_buffer, const int8_t* rowwise_buffer, const size_t row_count, const size_t col_count, const size_t* col_widths, const size_t row_size) {
const auto thread_index =
threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
if (thread_index >= row_count) {
return;
}
auto read_ptr = rowwise_buffer + thread_index * row_size;
auto col_base = columnar_buffer;
for (size_t i = 0; i < col_count; ++i) {
switch (col_widths[i]) {
case 8: {
int64_t* write_ptr = reinterpret_cast<int64_t*>(col_base) + thread_index;
*write_ptr = *reinterpret_cast<const int64_t*>(read_ptr);
} break;
case 4: {
int32_t* write_ptr = reinterpret_cast<int32_t*>(col_base) + thread_index;
*write_ptr = *reinterpret_cast<const int32_t*>(read_ptr);
} break;
default:;
}
col_base += col_widths[i] * row_count;
read_ptr += col_widths[i];  // WARN(miyu): No padding!!
}
}