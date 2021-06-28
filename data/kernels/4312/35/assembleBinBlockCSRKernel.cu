#include "includes.h"
__global__ void assembleBinBlockCSRKernel( const unsigned matrix_size, const float* diagonal_blks, const float* nondiagonal_blks, const int* csr_rowptr, const unsigned* blkrow_offset, float* JtJ_data ) {
const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
if(row_idx >= matrix_size) return;

//Now the query should all be safe
int data_offset = csr_rowptr[row_idx];
const auto blkrow_idx = row_idx / 6;
const auto inblk_offset = row_idx % 6;

//First fill the diagonal blks
for (auto k = 0; k < 6; k++, data_offset += bin_size) {
JtJ_data[data_offset] = diagonal_blks[36 * blkrow_idx + inblk_offset + 6 * k];
}

//Next fill the non-diagonal blks
auto Iij_begin = blkrow_offset[blkrow_idx];
const auto Iij_end = blkrow_offset[blkrow_idx + 1];
for (; Iij_begin < Iij_end; Iij_begin++) {
for (int k = 0; k < 6; k++, data_offset += bin_size) {
JtJ_data[data_offset] = nondiagonal_blks[36 * Iij_begin + inblk_offset + 6 * k];
}
}
}