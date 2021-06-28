#include "includes.h"
extern "C" {
}



#define TB 256
#define EPS 1e-4

__global__ void reconstruction_best_kernel( float *input, float *filtered_affine_model, float *filtered_best_output, int h, int w )
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
int size = h * w;
if (id < size) {
double out1 =
input[id + 2*size] * filtered_affine_model[id*12 + 0] + // A[0][0] +
input[id + size]   * filtered_affine_model[id*12 + 1] + // A[0][1] +
input[id]          * filtered_affine_model[id*12 + 2] + // A[0][2] +
filtered_affine_model[id*12 + 3]; //A[0][3];
double out2 =
input[id + 2*size] * filtered_affine_model[id*12 + 4] + //A[1][0] +
input[id + size]   * filtered_affine_model[id*12 + 5] + //A[1][1] +
input[id]          * filtered_affine_model[id*12 + 6] + //A[1][2] +
filtered_affine_model[id*12 + 7]; //A[1][3];
double out3 =
input[id + 2*size] * filtered_affine_model[id*12 + 8] + //A[2][0] +
input[id + size]   * filtered_affine_model[id*12 + 9] + //A[2][1] +
input[id]          * filtered_affine_model[id*12 + 10] + //A[2][2] +
filtered_affine_model[id*12 + 11]; // A[2][3];

filtered_best_output[id] = out1;
filtered_best_output[id + size] = out2;
filtered_best_output[id + 2*size] = out3;
}
return ;
}