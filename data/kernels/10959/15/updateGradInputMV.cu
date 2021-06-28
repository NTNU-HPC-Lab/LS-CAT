#include "includes.h"
__global__ void updateGradInputMV(const float* score, const float* weight, const float* mapping, const float* n_class_in_cluster, const float* class_start_indices, const float* target, const long gradInput_stride0, const long weight_stride0, const long score_stride0, int input_size, float* gradInput) {
// align input and score to current sample in minibatch
gradInput += gradInput_stride0 * blockIdx.y;
score += score_stride0 * blockIdx.y;

// get the indices corresponding the the target
const int itarget = (int)(target[blockIdx.y] - 0.5f); // - 0.5 : 1based->0
const int cluster_target = (int)(mapping[2*itarget] - 0.5f);
const int iclass_start = (int)(class_start_indices[cluster_target] + 0.5f);
const int cluster_size = (int)(n_class_in_cluster[cluster_target] + 0.5f);

// get the bias and weight of the target cluster + correct line
const int colIdx = blockIdx.x * MV2_NLINES + threadIdx.x;
const int nColParallel = gridDim.x * MV2_NLINES;

//   loop over lines
weight += weight_stride0 * iclass_start;
for (int icol = colIdx; icol < input_size; icol += nColParallel) {
const float* weight0 = weight + icol;
//   map
register float tmp = 0.f;
for (int i = 0; i < cluster_size; ++i)
tmp += score[i] * weight0[weight_stride0 * i];
gradInput[icol] = tmp;
}
}