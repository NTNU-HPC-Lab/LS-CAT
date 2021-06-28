#include "includes.h"
__global__ void sort_boxes_by_indexes_kernel(float* filtered_box, int* filtered_dir, float* box_for_nms, int* indexes, int filter_count, float* sorted_filtered_boxes, int* sorted_filtered_dir, float* sorted_box_for_nms, const int NUM_BOX_CORNERS, const int NUM_OUTPUT_BOX_FEATURE)
{
int tid = threadIdx.x + blockIdx.x * blockDim.x;
if(tid < filter_count)
{
int sort_index = indexes[tid];
sorted_filtered_boxes[tid*NUM_OUTPUT_BOX_FEATURE + 0] = filtered_box[sort_index*NUM_OUTPUT_BOX_FEATURE + 0];
sorted_filtered_boxes[tid*NUM_OUTPUT_BOX_FEATURE + 1] = filtered_box[sort_index*NUM_OUTPUT_BOX_FEATURE + 1];
sorted_filtered_boxes[tid*NUM_OUTPUT_BOX_FEATURE + 2] = filtered_box[sort_index*NUM_OUTPUT_BOX_FEATURE + 2];
sorted_filtered_boxes[tid*NUM_OUTPUT_BOX_FEATURE + 3] = filtered_box[sort_index*NUM_OUTPUT_BOX_FEATURE + 3];
sorted_filtered_boxes[tid*NUM_OUTPUT_BOX_FEATURE + 4] = filtered_box[sort_index*NUM_OUTPUT_BOX_FEATURE + 4];
sorted_filtered_boxes[tid*NUM_OUTPUT_BOX_FEATURE + 5] = filtered_box[sort_index*NUM_OUTPUT_BOX_FEATURE + 5];
sorted_filtered_boxes[tid*NUM_OUTPUT_BOX_FEATURE + 6] = filtered_box[sort_index*NUM_OUTPUT_BOX_FEATURE + 6];

sorted_filtered_dir[tid] = filtered_dir[sort_index];


sorted_box_for_nms[tid*NUM_BOX_CORNERS + 0] = box_for_nms[sort_index*NUM_BOX_CORNERS + 0];
sorted_box_for_nms[tid*NUM_BOX_CORNERS + 1] = box_for_nms[sort_index*NUM_BOX_CORNERS + 1];
sorted_box_for_nms[tid*NUM_BOX_CORNERS + 2] = box_for_nms[sort_index*NUM_BOX_CORNERS + 2];
sorted_box_for_nms[tid*NUM_BOX_CORNERS + 3] = box_for_nms[sort_index*NUM_BOX_CORNERS + 3];
}
}