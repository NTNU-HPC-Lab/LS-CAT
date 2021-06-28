#include "includes.h"
__device__ void recover_results(short *results, const int search_depth, const int total_dl_matrix_row_num) {
for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
if (results[i] == search_depth) {
results[i] = 0;
}
}
}
__global__ void recover_results(int *results, const int search_depth, const int total_dl_matrix_row_num) {
for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
if (results[i] == search_depth) {
results[i] = 0;
}
}
}