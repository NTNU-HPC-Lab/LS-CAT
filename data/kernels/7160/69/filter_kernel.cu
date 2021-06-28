#include "includes.h"
__global__ void filter_kernel(const float* box_preds, const float* cls_preds, const float* dir_preds, const int* anchor_mask, const float* dev_anchors_px, const float* dev_anchors_py, const float* dev_anchors_pz, const float* dev_anchors_dx, const float* dev_anchors_dy, const float* dev_anchors_dz, const float* dev_anchors_ro, float* filtered_box, float* filtered_score, int* filtered_dir, float* box_for_nms, int* filter_count, const float FLOAT_MIN, const float FLOAT_MAX, const float score_threshold, const int NUM_BOX_CORNERS, const int NUM_OUTPUT_BOX_FEATURE)
{
// boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
int tid = threadIdx.x + blockIdx.x * blockDim.x;
//sigmoid funciton
float score = 1/(1+expf(-cls_preds[tid]));
if(anchor_mask[tid] == 1 && score > score_threshold)
{
int counter = atomicAdd(filter_count, 1);
float za = dev_anchors_pz[tid] + dev_anchors_dz[tid]/2;

//decode network output
float diagonal = sqrtf(dev_anchors_dx[tid]*dev_anchors_dx[tid] + dev_anchors_dy[tid]*dev_anchors_dy[tid]);
float box_px = box_preds[tid*NUM_OUTPUT_BOX_FEATURE + 0] * diagonal + dev_anchors_px[tid];
float box_py = box_preds[tid*NUM_OUTPUT_BOX_FEATURE + 1] * diagonal + dev_anchors_py[tid];
float box_pz = box_preds[tid*NUM_OUTPUT_BOX_FEATURE + 2] * dev_anchors_dz[tid] + za;
float box_dx = expf(box_preds[tid*NUM_OUTPUT_BOX_FEATURE + 3]) * dev_anchors_dx[tid];
float box_dy = expf(box_preds[tid*NUM_OUTPUT_BOX_FEATURE + 4]) * dev_anchors_dy[tid];
float box_dz = expf(box_preds[tid*NUM_OUTPUT_BOX_FEATURE + 5]) * dev_anchors_dz[tid];
float box_ro = box_preds[tid*NUM_OUTPUT_BOX_FEATURE + 6] + dev_anchors_ro[tid];

box_pz = box_pz - box_dz/2;

filtered_box[counter*NUM_OUTPUT_BOX_FEATURE + 0] = box_px;
filtered_box[counter*NUM_OUTPUT_BOX_FEATURE + 1] = box_py;
filtered_box[counter*NUM_OUTPUT_BOX_FEATURE + 2] = box_pz;
filtered_box[counter*NUM_OUTPUT_BOX_FEATURE + 3] = box_dx;
filtered_box[counter*NUM_OUTPUT_BOX_FEATURE + 4] = box_dy;
filtered_box[counter*NUM_OUTPUT_BOX_FEATURE + 5] = box_dz;
filtered_box[counter*NUM_OUTPUT_BOX_FEATURE + 6] = box_ro;
filtered_score[counter] = score;

int direction_label;
if(dir_preds[tid*2 + 0] < dir_preds[tid*2 + 1])
{
direction_label = 1;
}
else
{
direction_label = 0;
}
filtered_dir[counter] = direction_label;

//convrt normal box(normal boxes: x, y, z, w, l, h, r) to box(xmin, ymin, xmax, ymax) for nms calculation
//First: dx, dy -> box(x0y0, x0y1, x1y0, x1y1)
float corners[NUM_3D_BOX_CORNERS_MACRO] = {float(-0.5*box_dx), float(-0.5*box_dy),
float(-0.5*box_dx), float( 0.5*box_dy),
float( 0.5*box_dx), float( 0.5*box_dy),
float( 0.5*box_dx), float(-0.5*box_dy)};

//Second: Rotate, Offset and convert to point(xmin. ymin, xmax, ymax)
float rotated_corners[NUM_3D_BOX_CORNERS_MACRO];
float offset_corners[NUM_3D_BOX_CORNERS_MACRO];
float sin_yaw = sinf(box_ro);
float cos_yaw = cosf(box_ro);
float xmin = FLOAT_MAX;
float ymin = FLOAT_MAX;
float xmax = FLOAT_MIN;
float ymax = FLOAT_MIN;
for(size_t i = 0; i < NUM_BOX_CORNERS; i++)
{
rotated_corners[i*2 + 0] = cos_yaw*corners[i*2 + 0] - sin_yaw*corners[i*2 + 1];
rotated_corners[i*2 + 1] = sin_yaw*corners[i*2 + 0] + cos_yaw*corners[i*2 + 1];

offset_corners[i*2 + 0] = rotated_corners[i*2 + 0] + box_px;
offset_corners[i*2 + 1] = rotated_corners[i*2 + 1] + box_py;

xmin = fminf(xmin, offset_corners[i*2 + 0]);
ymin = fminf(ymin, offset_corners[i*2 + 1]);
xmax = fmaxf(xmin, offset_corners[i*2 + 0]);
ymax = fmaxf(ymax, offset_corners[i*2 + 1]);
}
// box_for_nms(num_box, 4)
box_for_nms[counter*NUM_BOX_CORNERS + 0] = xmin;
box_for_nms[counter*NUM_BOX_CORNERS + 1] = ymin;
box_for_nms[counter*NUM_BOX_CORNERS + 2] = xmax;
box_for_nms[counter*NUM_BOX_CORNERS + 3] = ymax;

}
}