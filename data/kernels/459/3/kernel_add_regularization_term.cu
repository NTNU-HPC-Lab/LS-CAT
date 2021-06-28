#include "includes.h"
__global__ void kernel_add_regularization_term(double     * d_input_vector, int           dimension, double       regularization_parameter, double     * d_rv)
{
if (threadIdx.x == 0) {
double sum = 0;
for (int i = 1; i < dimension; ++i) {
sum += 0.5 * d_input_vector[i] * d_input_vector[i] * regularization_parameter;
}
*d_rv += sum;
}
}