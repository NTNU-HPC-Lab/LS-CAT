#include "includes.h"
__global__ void kEltwiseLogregCost(float* predmap, float* indmap, float*indlogpred, float* correctprobs, int numCases, int numTasks, int per_thread_case) {
const int task_id = blockIdx.x;
const int start_tx = threadIdx.x * per_thread_case;
const int end_tx = min(start_tx + per_thread_case, numCases);
const float EPSILON=1e-20; // Minimum value allowed, avoid log( 0 )
if (task_id >= numTasks) {
return;
}
for (int c_id = start_tx; c_id < end_tx; ++c_id) {
int pos = task_id * numCases + c_id;
float t = __fdividef(1.0f, 1.0f + __expf(-predmap[ pos ]));
if (indmap[pos] == 1) {
t = fmaxf(t, EPSILON);
indlogpred[pos] = __logf(t);
correctprobs[pos] = t;
} else {
t = 1-t;
t = fmaxf(t, EPSILON);
indlogpred[pos] = __logf(t);
correctprobs[pos] = t;
}
}
}