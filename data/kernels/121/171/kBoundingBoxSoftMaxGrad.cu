#include "includes.h"
__global__ void kBoundingBoxSoftMaxGrad( float* mat, int* bbox, int* label, int* seg, float* indices, float* width_offset, float* height_offset, int size, int width, int height, int depth, float scale_width, float scale_height, float* grad) {
const unsigned int len = width * height * depth * size;
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
int ind, image_id, source_depth, x1, y1, x2, y2, start,
end, src_image_id, num_bboxes, num_bboxes_of_this_depth, box_id, inside;
float source_x, source_y;
for (unsigned int i = idx; i < len; i += numThreads) {
ind = i;
image_id = ind % size; ind /= size;
source_x = scale_width * (ind % width); ind /= width;
source_y = scale_height * (ind % height); ind /= height;
source_depth = ind % depth;
src_image_id = (int)indices[image_id];

start = seg[src_image_id];
end = seg[src_image_id + 1];
num_bboxes = 0;
num_bboxes_of_this_depth = 0;
for (box_id = start; box_id < end; box_id++) {
x1 = bbox[box_id << 2] - width_offset[image_id];
y1 = bbox[(box_id << 2) + 1] - height_offset[image_id];
x2 = bbox[(box_id << 2) + 2] - width_offset[image_id];
y2 = bbox[(box_id << 2) + 3] - height_offset[image_id];
inside = (source_x >= x1 && source_x <= x2 && source_y >= y1 && source_y <= y2) ? 1:0;
num_bboxes += inside;
num_bboxes_of_this_depth += (inside == 1 && label[box_id] == source_depth) ? 1: 0;
}
grad[i] = mat[i] - ((num_bboxes > 0) ? ((float)num_bboxes_of_this_depth / num_bboxes) : (source_depth == 0 ? 1:0));
}
}