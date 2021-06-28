#include "includes.h"
extern "C" {
}

#define IDX2C(i, j, ld) ((j)*(ld)+(i))
#define SQR(x)      ((x)*(x))                        // x^2

__global__ void assemble_tensors(double const* tensor_input, double* tensors, int tensor_input_elements){
int tensor_matrix_offset = blockIdx.x * TENSOR_DIMENSIONS * TENSOR_DIMENSIONS;
int input_matrix_offset = blockIdx.x * tensor_input_elements;
tensors[tensor_matrix_offset + 0] = tensor_input[input_matrix_offset + 0];
tensors[tensor_matrix_offset + 1] = tensor_input[input_matrix_offset + 1];
tensors[tensor_matrix_offset + 2] = tensor_input[input_matrix_offset + 3];
tensors[tensor_matrix_offset + 3] = tensor_input[input_matrix_offset + 1];
tensors[tensor_matrix_offset + 4] = tensor_input[input_matrix_offset + 2];
tensors[tensor_matrix_offset + 5] = tensor_input[input_matrix_offset + 4];
tensors[tensor_matrix_offset + 6] = tensor_input[input_matrix_offset + 3];
tensors[tensor_matrix_offset + 7] = tensor_input[input_matrix_offset + 4];
tensors[tensor_matrix_offset + 8] = tensor_input[input_matrix_offset + 5];
}