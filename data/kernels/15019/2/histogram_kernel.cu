#include "includes.h"
__global__ void histogram_kernel(float* magnitude, float* phase, float* histograms, int input_width, int input_height, int cell_grid_width, int cell_grid_height, int magnitude_step, int phase_step, int histograms_step, int cell_width, int cell_height, int num_bins)
{
//TODO: make the buffer sizes dependent on an input or template parameter.
// Each thread block needs to store intermediate results for 64 gradients
// and also 8 different histograms, each with 9 bins.
__shared__ int s_lbin_pos[64];
__shared__ float s_lbin[64];
__shared__ int s_rbin_pos[64];
__shared__ float s_rbin[64];
__shared__ float s_hist[9 * 8];

// The columns of the image are mapped to the first dimension of the block
// grid and the first dimension of the thread block.
int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
// If current position is outside the image, stop here
if(pixel_x >= input_width)
{
return;
}
// The columns of the image are mapped to the second dimension of the block
// grid and the second dimension of the thread block.
int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
// If current position is outside the image, stop here
if(pixel_y >= input_height)
{
return;
}

// Each row has magnitude_step size
int mag_pixel_idx = pixel_y * magnitude_step + pixel_x;
// Each row has phase_step size
int phase_pixel_idx = pixel_y * phase_step + pixel_x;

// The phase was previously normalized to [0,1]
float bin_size = 1.0f / (float)num_bins;
// By dividing by the bin size and taking the integer part, you find out
// inside which bin the gradient is at. If it's greater than the middle of the bin
// it will be divided between this one and the next, if it's lesser it will
// be divided between this and the previous one. By subtracting 0.5 before
// taking the integer part, the division will always be between this bin and
// the next.
int left_bin = (int)floor((phase[phase_pixel_idx] / bin_size) - 0.5f);
// The result of the previous operation might be negative. If so, the next
// bit fixes that. Otherwise that changes nothing.
left_bin = (left_bin + num_bins) % num_bins;
// Take the next bin as the right bin.
// If the left bin is the last one, this will be outside range. Wait a bit
// before taking the remainder, because this value needs to be used in the
// formula below.
int right_bin = (left_bin + 1);
// Calculate the distance between the gradient phase and the limit between
// the left and right bins. Normalized by the bin size, the limit is equal
// to the right bin identifier.
float delta = (phase[phase_pixel_idx] / bin_size) - right_bin;
if(delta < -0.5)
{
delta += num_bins;
}
//Fix range for right_bin now
right_bin = right_bin % num_bins;

// Store the bin positions and amounts for each bin on shared buffers.
s_lbin_pos[threadIdx.x] = left_bin;
s_lbin[threadIdx.x] = (0.5 - delta) * magnitude[mag_pixel_idx];
s_rbin_pos[threadIdx.x] = right_bin;
s_rbin[threadIdx.x] = (0.5 + delta) * magnitude[mag_pixel_idx];

// Wait for other threads.
__syncthreads();

// Initialize histograms shared buffer.
s_hist[threadIdx.x] = 0.0f;
if(threadIdx.x < 8)
{
s_hist[threadIdx.x + 64] = 0.0f;
}

int cell_y = pixel_y / cell_height;

// Each partial histogram will be calculated by only one thread.
if(threadIdx.x < 8)
{
int s_hist_idx = 9 * threadIdx.x;
for(int i = 1; i < 8; ++i)
{
s_hist[s_hist_idx + s_lbin_pos[8 * threadIdx.x + i]] += s_lbin[8
* threadIdx.x + i];
s_hist[s_hist_idx + s_rbin_pos[8 * threadIdx.x + i]] += s_rbin[8
* threadIdx.x + i];
}
}

// Wait until all threads finish.
__syncthreads();

// Add to the complete histogram sum using atomic operations.
int out_idx = cell_y * histograms_step + threadIdx.x;
atomicAdd(&(histograms[out_idx]), s_hist[threadIdx.x]);

if(threadIdx.x < 8)
{
atomicAdd(&(histograms[out_idx + 64]), s_hist[threadIdx.x + 64]);
}
}