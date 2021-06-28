#include "includes.h"
__global__ void buildCompactedIndexKernel( const unsigned* valid_indicator, const unsigned table_size, unsigned* compacted_index ) {
const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
if(idx < table_size) {
unsigned offset = 0xffffffffu;
if(valid_indicator[idx] > 0) {
offset = compacted_index[idx] - 1;
}
compacted_index[idx] = offset;
}
}