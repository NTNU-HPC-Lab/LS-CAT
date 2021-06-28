#include "includes.h"
__device__ __host__ static inline uint8_t xnor_bit1(uint8_t a, uint8_t b) {
return ~(a^b) & 0b1;
}
__device__ __host__ static inline unsigned char get_bit(unsigned char const*const src, size_t index) {
size_t src_i = index / 8;
int src_shift = index % 8;
unsigned char val = (src[src_i] & (1 << src_shift)) > 0;
//unsigned char val = (src[src_i] & (1 << (8 - src_shift))) > 0;
return val;
}
__global__ void convolve_bin_gpu_kernel(float *input, float *weights, float *output, int in_w, int in_h, int in_c, int n, int size, int pad, int new_lda, float *mean_arr_gpu)
{
int index = blockIdx.x*blockDim.x + threadIdx.x;

int fil;
// filter index
//for (fil = 0; fil < n; ++fil)
int chan, y, x, f_y, f_x;
// channel index
//for (chan = 0; chan < in_c; ++chan)
// input - y
//for (y = 0; y < in_h; ++y)
// input - x
//for (x = 0; x < in_w; ++x)
x = index % in_w;
int index2 = index / in_w;
y = index2 % in_h;
fil = index2 / in_h;
//if (fil < n)    // (1-6 for one BLOCK)
{
//float mean_val = mean_arr_gpu[fil];
int const output_index = fil*in_w*in_h + y*in_w + x;
int sum = 0;
int good_val = 0;

int min_index = blockIdx.x*blockDim.x;
int min_fil = (min_index / in_w) / in_h;
int max_index = (blockIdx.x+1)*blockDim.x - 1;
int max_fil = (max_index / in_w) / in_h;

__shared__ uint32_t weights_shared[3*3*1024*6/32 + 1];  // 7 KB (6 filters) - use (new_lda) for size calculation
//const int weights_size = size*size*in_c/8;
const int weights_size = size*size*in_c / 32 + 1;

for (int tmp_fil = min_fil; tmp_fil <= max_fil; tmp_fil++) {
for (int s = threadIdx.x; s < weights_size; s += blockDim.x) {
//weights_shared[s + (tmp_fil - min_fil)*new_lda / 8] = ((uint8_t *)weights)[tmp_fil*new_lda / 8 + s];
weights_shared[s + (tmp_fil - min_fil)*new_lda/32] = ((uint32_t *)weights)[tmp_fil*new_lda / 32 + s];
}
}
__syncthreads();

for (chan = 0; chan < in_c; ++chan)
{
//int const weights_pre_index = fil*in_c*size*size + chan*size*size;
//int const weights_pre_index = fil*new_lda + chan*size*size;
int const input_pre_index = chan*in_w*in_h;

__shared__ uint32_t input_shared[416*416/32 + 1];   // 21.2 KB bytes (for input size 832x832)
const int input_shared_size = in_w*in_h / 32 + 1;
const int add_input_index = input_pre_index % 32;
__syncthreads();    // why??? but is required

for (int s = threadIdx.x; s < input_shared_size; s += blockDim.x) {
input_shared[s] = ((uint32_t *)input)[input_pre_index / 32 + s];
}
__syncthreads();

/*
__shared__ uint8_t input_shared[208 * 208 / 8 + 1];   // 5.4 KB bytes (for input size 416x416)
const int input_shared_size = in_w*in_h / 8 + 1;
const int add_input_index = input_pre_index % 8;
__syncthreads();

for (int s = threadIdx.x; s < input_shared_size; s += blockDim.x) {
((uint8_t *)input_shared)[s] = ((uint8_t *)input)[input_pre_index / 8 + s];
}
__syncthreads();
*/
//int src_index = -1;
//uint32_t input_byte;

if (fil < n)    // (1-6 for one BLOCK)
{
// filter - y
for (f_y = 0; f_y < size; ++f_y)
{
int input_y = y + f_y - pad;
// filter - x
for (f_x = 0; f_x < size; ++f_x)
{
int input_x = x + f_x - pad;
if (input_y < 0 || input_x < 0 || input_y >= in_h || input_x >= in_w) continue;

//int input_index = input_pre_index + input_y*in_w + input_x;
//int weights_index = weights_pre_index + f_y*size + f_x;
//int weights_index = fil*in_c*size*size + chan*size*size + f_y*size + f_x;
//int weights_index = fil*new_lda + chan*size*size + f_y*size + f_x;

//uint8_t in_bit = get_bit((uint8_t *)input, input_index);
//uint8_t w_bit = get_bit((uint8_t *)weights, weights_index);

//int weights_index = fil*in_c*size*size + chan*size*size + f_y*size + f_x;
int weights_shared_index = (fil - min_fil)*new_lda + chan*size*size + f_y*size + f_x;
//uint8_t in_bit = get_bit((uint8_t *)weights_shared, weights_shared_index);
uint8_t w_bit = get_bit((uint8_t *)weights_shared, weights_shared_index);

//int input_index = input_pre_index + input_y*in_w + input_x;
int input_shared_index = /*input_pre_index +*/ input_y*in_w + input_x + add_input_index;
uint8_t in_bit = get_bit((uint8_t *)input_shared, input_shared_index);
/*
int new_src_index = input_shared_index / 32;
int src_shift = input_shared_index % 32;
//if (new_src_index != src_index)
{
src_index = new_src_index;
input_byte = ((uint32_t *)input_shared)[src_index];
}
uint8_t in_bit = (input_byte & (1 << src_shift)) >> src_shift;
*/

int res = xnor_bit1(in_bit, w_bit);
sum += res;
good_val++;

//sum += input[input_index] *weights[weights_index];

}
}
}
// l.output[filters][width][height] +=
//        state.input[channels][width][height] *
//        l.weights[filters][channels][filter_width][filter_height];
//output[output_index] += sum;
}
sum = sum - (good_val - sum);
//output[output_index] = sum * mean_arr_gpu[fil]; // atoimcAdd for inter-BLOCK sum
atomicAdd(&output[output_index], sum * mean_arr_gpu[fil]);
}

}