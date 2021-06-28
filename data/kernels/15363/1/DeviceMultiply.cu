#include "includes.h"
__global__ void DeviceMultiply(double* left, double* right, double* result, int left_rows, int left_cols, int right_cols) {
int i = threadIdx.y;
int j = threadIdx.x;
int x_stride = blockDim.x;
int y_stride = blockDim.y;
__shared__ double temp[16][16];
for (int y_offset = 0; i + y_offset < left_rows; y_offset += y_stride) {
for (int x_offset = 0; j + x_offset < right_cols; x_offset += x_stride) {
temp[i][j] = 0.0;
for (int k = 0; k < left_cols; ++k) {
int left_idx = (y_offset + i) * left_cols + k;
int right_idx = k * right_cols + x_offset + j;
temp[i][j] += left[left_idx] * right[right_idx];
}
int result_idx = (y_offset + i) * right_cols + x_offset + j;
result[result_idx] = temp[i][j];
}
}
}