#include "includes.h"
__global__ void kEltwiseL2SVMCost(float* ydata, float* ldata, float* pre_grad, float* all_cost, float a, float b, int numCases, int numTasks, int per_thread_case) {
const int task_id = blockIdx.x;
const int start_tx = threadIdx.x * per_thread_case;
const int end_tx = min(start_tx + per_thread_case, numCases);
if (task_id >= numTasks) {
return;
}
for (int c_id = start_tx; c_id < end_tx; ++c_id) {
int pos = task_id * numCases + c_id;
float tmp = fmaxf(a - ydata[pos] * (ldata[pos] - b), 0);
pre_grad[pos] = tmp;
all_cost[pos] = tmp*tmp;
}
}