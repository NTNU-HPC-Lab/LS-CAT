#include "includes.h"


static const int NTHREADS = 32;





__global__ void cunn_ClassNLLCriterion_updateGradInput_kernel1( float* gradInput, float* weights, float* target, float* total_weight, int size_average, int n_classes)
{
if (*total_weight <= 0) {
return;
}
float norm = size_average ? (1.0f / *total_weight) : 1.0f;
int t = (int)*target - 1;
if (t >= 0 && t < n_classes) {
gradInput[t] = -(weights ? weights[t] : 1.0f) * norm;
}
}