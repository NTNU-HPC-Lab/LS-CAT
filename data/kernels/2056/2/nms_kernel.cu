#include "includes.h"
__global__ void nms_kernel( const int num_per_thread, const float threshold, const int num_detections, const int *indices, float *scores, const float *classes, const float4 *boxes) {

// Go through detections by descending score
for (int m = 0; m < num_detections; m++) {
for (int n = 0; n < num_per_thread; n++) {
int i = threadIdx.x * num_per_thread + n;
if (i < num_detections && m < i && scores[m] > 0.0f) {
int idx = indices[i];
int max_idx = indices[m];
int icls = classes[idx];
int mcls = classes[max_idx];
if (mcls == icls) {
float4 ibox = boxes[idx];
float4 mbox = boxes[max_idx];
float x1 = max(ibox.x, mbox.x);
float y1 = max(ibox.y, mbox.y);
float x2 = min(ibox.z, mbox.z);
float y2 = min(ibox.w, mbox.w);
float w = max(0.0f, x2 - x1 + 1);
float h = max(0.0f, y2 - y1 + 1);
float iarea = (ibox.z - ibox.x + 1) * (ibox.w - ibox.y + 1);
float marea = (mbox.z - mbox.x + 1) * (mbox.w - mbox.y + 1);
float inter = w * h;
float overlap = inter / (iarea + marea - inter);
if (overlap > threshold) {
scores[i] = 0.0f;
}
}
}
}

// Sync discarded detections
__syncthreads();
}
}