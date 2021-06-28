#include "includes.h"
__global__ void updateOutputWithTargetMV(const float* input, const float* weight, const float* bias, const float* mapping, const float* n_class_in_cluster, const float* class_start_indices, const float* target, const long input_stride0, const long weight_stride0, const long score_stride0, long input_size, float* score) {
__shared__ float buffer[MV_BUFFER_SIZE];
// align input and score to current sample in minibatch
input += input_stride0 * blockIdx.y;
score += score_stride0 * blockIdx.y;

// get the indices corresponding the the target
const int itarget = (int)(target[blockIdx.y] - 0.5f); // - 0.5 : 1based->0
const int cluster_target = (int)(mapping[2*itarget] - 0.5f);
const int iclass_start = (int)(class_start_indices[cluster_target] + 0.5f);
const int cluster_size = (int)(n_class_in_cluster[cluster_target] + 0.5f);

// get the bias and weight of the target cluster + correct line
const int lineIdx = blockIdx.x;
const int nLinesParallel = gridDim.x;

// do matrix vector multiply :
const int tidxx = threadIdx.x;
//   loop over lines
for (int iline = lineIdx; iline < cluster_size; iline += nLinesParallel) {
const float* weight0 = weight + weight_stride0 * (iclass_start + iline);
//   map
__syncthreads();
register float tmp = 0.f;
for (int i = tidxx; i < input_size; i += MV_BUFFER_SIZE)
tmp += input[i] * weight0[i];
buffer[tidxx] = tmp;
//   reduce
/*
for (unsigned int stride = MV_BUFFER_SIZE >> 1; stride > 0; stride >>= 1) {
__syncthreads();
if (tidxx < stride)
buffer[tidxx] += buffer[tidxx+stride];
}
if (tidxx == 0)
score[iline] = buffer[0] + bias[iclass_start + iline];
*/
tmp = 0.f;
__syncthreads();
if (tidxx < MV_BUFFER_SIZE / MV_N_REDUCE) {
for (int i = tidxx * MV_N_REDUCE; i < (tidxx + 1) * MV_N_REDUCE; ++i)
tmp += buffer[i];
buffer[tidxx] = tmp;
}
__syncthreads();
// store result
if (tidxx == 0) {
tmp = buffer[0];
#pragma unroll
for (int i = 1; i < MV_BUFFER_SIZE / MV_N_REDUCE; ++i)
tmp += buffer[i];
score[iline] = tmp + bias[iclass_start + iline];
}
}
}