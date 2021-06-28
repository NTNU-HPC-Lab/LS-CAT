#include "includes.h"
__global__ void box_encode_kernel(float *targets_dx, float *targets_dy, float *targets_dw, float *targets_dh, float4 *boxes, float4 *anchors, float wx, float wy, float ww, float wh, size_t gt, size_t idxJump) {

int idx = blockIdx.x * blockDim.x + threadIdx.x;
size_t row_offset;
float anchors_x1, anchors_x2, anchors_y1, anchors_y2,
boxes_x1, boxes_x2, boxes_y1, boxes_y2, ex_w, ex_h,
ex_ctr_x, ex_ctr_y, gt_w, gt_h, gt_ctr_x, gt_ctr_y;

for (int i = idx; i < gt; i += idxJump){
row_offset = i;
anchors_x1 = anchors[row_offset].x;
anchors_y1 = anchors[row_offset].y;
anchors_x2 = anchors[row_offset].z;
anchors_y2 = anchors[row_offset].w;

boxes_x1 = boxes[row_offset].x;
boxes_y1 = boxes[row_offset].y;
boxes_x2 = boxes[row_offset].z;
boxes_y2 = boxes[row_offset].w;

ex_w = anchors_x2 - anchors_x1 + 1;
ex_h = anchors_y2 - anchors_y1 + 1;
ex_ctr_x = anchors_x1 + 0.5 * ex_w;
ex_ctr_y = anchors_y1 + 0.5 * ex_h;

gt_w = boxes_x2 - boxes_x1 + 1;
gt_h = boxes_y2 -  boxes_y1 + 1;
gt_ctr_x = boxes_x1 + 0.5 * gt_w;
gt_ctr_y = boxes_y1 + 0.5 * gt_h;

targets_dx[i] = wx * (gt_ctr_x - ex_ctr_x) / ex_w;
targets_dy[i] = wy * (gt_ctr_y - ex_ctr_y) / ex_h;
targets_dw[i] = ww * log(gt_w / ex_w);
targets_dh[i] = wh * log(gt_h / ex_h);
}

}