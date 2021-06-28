#include "includes.h"

using namespace std;




__global__ void multiplyDigits(char* d_str1, char* d_str2, int* d_matrix, int str1_len, int str2_len) {
int row = blockDim.y * blockIdx.x + threadIdx.y;
int col = blockDim.x * blockIdx.y + threadIdx.x;

int idx = row * str1_len + (col + (str2_len * row)) + 1 + (row);

d_matrix[idx] = (d_str2[row] - '0') * (d_str1[col] - '0');
}