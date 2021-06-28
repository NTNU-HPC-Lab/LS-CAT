#include "includes.h"
__global__ void sort_boxes_by_indexes_kernel( float* filtered_box, int* filtered_label, int* filtered_dir, float* box_for_nms, int* indexes, int filter_count, float* sorted_filtered_boxes, int* sorted_filtered_label, int* sorted_filtered_dir, float* sorted_box_for_nms, const int num_box_corners, const int num_output_box_feature) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
if (tid < filter_count) {
int sort_index = indexes[tid];
sorted_filtered_boxes[tid * num_output_box_feature + 0] =
filtered_box[sort_index * num_output_box_feature + 0];
sorted_filtered_boxes[tid * num_output_box_feature + 1] =
filtered_box[sort_index * num_output_box_feature + 1];
sorted_filtered_boxes[tid * num_output_box_feature + 2] =
filtered_box[sort_index * num_output_box_feature + 2];
sorted_filtered_boxes[tid * num_output_box_feature + 3] =
filtered_box[sort_index * num_output_box_feature + 3];
sorted_filtered_boxes[tid * num_output_box_feature + 4] =
filtered_box[sort_index * num_output_box_feature + 4];
sorted_filtered_boxes[tid * num_output_box_feature + 5] =
filtered_box[sort_index * num_output_box_feature + 5];
sorted_filtered_boxes[tid * num_output_box_feature + 6] =
filtered_box[sort_index * num_output_box_feature + 6];

sorted_filtered_label[tid] = filtered_label[sort_index];

sorted_filtered_dir[tid] = filtered_dir[sort_index];

sorted_box_for_nms[tid * num_box_corners + 0] =
box_for_nms[sort_index * num_box_corners + 0];
sorted_box_for_nms[tid * num_box_corners + 1] =
box_for_nms[sort_index * num_box_corners + 1];
sorted_box_for_nms[tid * num_box_corners + 2] =
box_for_nms[sort_index * num_box_corners + 2];
sorted_box_for_nms[tid * num_box_corners + 3] =
box_for_nms[sort_index * num_box_corners + 3];
}
}