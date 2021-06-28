#include "includes.h"
/*!
* Copyright (c) 2017 Microsoft
* Licensed under The MIT License [see LICENSE for details]
* \file deformable_psroi_pooling.cu
* \brief
* \author Yi Li, Guodong Zhang, Jifeng Dai
*/
/***************** Adapted by Charles Shang *********************/

#define CUDA_KERNEL_LOOP(i, n)                        \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
i < (n);                                       \
i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
__device__ float bilinear_interp( const float *data, const float x, const float y, const int width, const int height)
{
int x1 = floor(x);
int x2 = ceil(x);
int y1 = floor(y);
int y2 = ceil(y);
float dist_x = (float)(x - x1);
float dist_y = (float)(y - y1);
float value11 = data[y1 * width + x1];
float value12 = data[y2 * width + x1];
float value21 = data[y1 * width + x2];
float value22 = data[y2 * width + x2];
float value = (1 - dist_x) * (1 - dist_y) * value11 + (1 - dist_x) * dist_y * value12 + dist_x * (1 - dist_y) * value21 + dist_x * dist_y * value22;
return value;
}
__device__ double bilinear_interp( const double *data, const double x, const double y, const int width, const int height)
{
int x1 = floor(x);
int x2 = ceil(x);
int y1 = floor(y);
int y2 = ceil(y);
double dist_x = (double)(x - x1);
double dist_y = (double)(y - y1);
double value11 = data[y1 * width + x1];
double value12 = data[y2 * width + x1];
double value21 = data[y1 * width + x2];
double value22 = data[y2 * width + x2];
double value = (1 - dist_x) * (1 - dist_y) * value11 + (1 - dist_x) * dist_y * value12 + dist_x * (1 - dist_y) * value21 + dist_x * dist_y * value22;
return value;
}
__global__ void DeformablePSROIPoolForwardKernel( const int count, const double *bottom_data, const double spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const double *bottom_rois, const double *bottom_trans, const int no_trans, const double trans_std, const int sample_per_part, const int output_dim, const int group_size, const int part_size, const int num_classes, const int channels_each_class, double *top_data, double *top_count)
{
CUDA_KERNEL_LOOP(index, count)
{
// The output is in order (n, ctop, ph, pw)
int pw = index % pooled_width;
int ph = (index / pooled_width) % pooled_height;
int ctop = (index / pooled_width / pooled_height) % output_dim;
int n = index / pooled_width / pooled_height / output_dim;

// [start, end) interval for spatial sampling
const double *offset_bottom_rois = bottom_rois + n * 5;
int roi_batch_ind = offset_bottom_rois[0];
double roi_start_w = (double)(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
double roi_start_h = (double)(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
double roi_end_w = (double)(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
double roi_end_h = (double)(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;

// Force too small ROIs to be 1x1
double roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
double roi_height = max(roi_end_h - roi_start_h, 0.1);

// Compute w and h at bottom
double bin_size_h = roi_height / (double)(pooled_height);
double bin_size_w = roi_width / (double)(pooled_width);

double sub_bin_size_h = bin_size_h / (double)(sample_per_part);
double sub_bin_size_w = bin_size_w / (double)(sample_per_part);

int part_h = floor((double)(ph) / pooled_height * part_size);
int part_w = floor((double)(pw) / pooled_width * part_size);
int class_id = ctop / channels_each_class;
double trans_x = no_trans ? (double)(0) : bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w] * trans_std;
double trans_y = no_trans ? (double)(0) : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w] * trans_std;

double wstart = (double)(pw)*bin_size_w + roi_start_w;
wstart += trans_x * roi_width;
double hstart = (double)(ph)*bin_size_h + roi_start_h;
hstart += trans_y * roi_height;

double sum = 0;
int count = 0;
int gw = floor((double)(pw)*group_size / pooled_width);
int gh = floor((double)(ph)*group_size / pooled_height);
gw = min(max(gw, 0), group_size - 1);
gh = min(max(gh, 0), group_size - 1);

const double *offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
for (int ih = 0; ih < sample_per_part; ih++)
{
for (int iw = 0; iw < sample_per_part; iw++)
{
double w = wstart + iw * sub_bin_size_w;
double h = hstart + ih * sub_bin_size_h;
// bilinear interpolation
if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5)
{
continue;
}
w = min(max(w, 0.), width - 1.);
h = min(max(h, 0.), height - 1.);
int c = (ctop * group_size + gh) * group_size + gw;
double val = bilinear_interp(offset_bottom_data + c * height * width, w, h, width, height);
sum += val;
count++;
}
}
top_data[index] = count == 0 ? (double)(0) : sum / count;
top_count[index] = count;
}
}