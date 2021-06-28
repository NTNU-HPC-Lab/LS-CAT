#include "includes.h"
/*
* This code is released into the public domain.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
* OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*/






///////////////////////////////////////////////////////////////////////////////////////////
// Definitions and helper utilities

// Block width for CUDA kernels
#define BW 128

#ifdef USE_GFLAGS

#ifndef _WIN32
#define gflags google
#endif
#else
// Constant versions of gflags
#define DEFINE_int32(flag, default_value, description) const int FLAGS_##flag = (default_value)
#define DEFINE_uint64(flag, default_value, description) const unsigned long long FLAGS_##flag = (default_value)
#define DEFINE_bool(flag, default_value, description) const bool FLAGS_##flag = (default_value)
#define DEFINE_double(flag, default_value, description) const double FLAGS_##flag = (default_value)
#define DEFINE_string(flag, default_value, description) const std::string FLAGS_##flag ((default_value))
#endif

/**
* Computes ceil(x / y) for integral nonnegative values.
*/
__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= batch_size)
return;

const int label_value = static_cast<int>(label[idx]);

// For each item in the batch, decrease the result of the label's value by 1
diff[idx * num_labels + label_value] -= 1.0f;
}