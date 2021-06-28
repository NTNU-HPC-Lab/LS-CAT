#include "includes.h"
__global__ void cunn_TemporalMaxPooling_updateGradInputKernelAtomic(float *gradInput, float *gradOutput, float *indices, int input_w, int input_n, int output_w, int kW, int dW) {
// Block idx is the batch index, thread idx + block idx y * MAX_THREADS is the time index
float *gradInput_data = gradInput + blockIdx.x * input_w * input_n + (
threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n * dW;
float *gradOutput_data = gradOutput + blockIdx.x * output_w * input_n + (
threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;
float *indices_data = indices + blockIdx.x * output_w * input_n + (
threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;

int feat = 0;

if (threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS < output_w) {
// For all features
for (feat = 0; feat < input_n; ++feat) {
atomicAdd(&gradInput_data[(int)indices_data[feat] * input_n + feat], gradOutput_data[feat]);
}
}
}