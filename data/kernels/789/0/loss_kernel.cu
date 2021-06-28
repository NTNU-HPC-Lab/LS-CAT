#include "includes.h"
__device__ float get_prediction(int factors, const float *p, const float *q, float user_bias, float item_bias, float global_bias) {
float pred = global_bias + user_bias + item_bias;
for (int f = 0; f < factors; f++)
pred += q[f]*p[f];
return pred;
}
__global__ void loss_kernel(int factors, int user_count, int item_count, const float * P, const float * Q, const int * indptr, const int * indices, const float * data, float * error, float * user_bias, float * item_bias, float global_bias) {

// One thread per user
int u = blockDim.x * blockIdx.x + threadIdx.x;
if(u < user_count) {
// Get this user's factors and bias
const float * p = &P[u * factors];
const float ub = user_bias[u];

// Loop over all items of user
for (int i = indptr[u]; i < indptr[u + 1]; ++i) {
int item_id = indices[i];
error[i] = data[i] - get_prediction(factors, p, &Q[item_id * factors], ub, item_bias[item_id], global_bias);
}
}
}