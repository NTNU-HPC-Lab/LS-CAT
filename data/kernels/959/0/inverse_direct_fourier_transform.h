
// Copyright 2019 Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2019 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifdef __cplusplus
extern "C" {
#endif

#ifndef INVERSE_DIRECT_FOURIER_TRANSFORM_H_
#define INVERSE_DIRECT_FOURIER_TRANSFORM_H_

#include <ctime>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

//define function for checking CUDA errors
#define CUDA_CHECK_RETURN(value) check_cuda_error_aux(__FILE__,__LINE__, #value, value)

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

// Speed of light
#ifndef C
	#define C 299792458.0
#endif

//Define struct for the configuration
typedef struct Config {
	int vis_count;
	const char *output_image_file;
	const char *vis_file;
	bool psf_enabled;
	double image_size;
	int render_size;
	int x_render_offset;
	int y_render_offset;
	double cell_size;
	double uv_scale;
	double frequency_hz;
	bool enable_right_ascension;
	int vis_batch_size;
	int gpu_num_threads_per_block_dimension;
} Config;

//Define struct for visibility coordinate
typedef struct Visibility {
	double u;
	double v;
	double w;
} Visibility;

//used for grid(image) and visibility intensity
typedef struct Complex {
	double real;
	double imaginary;
} Complex;


void init_config(Config *config);

void load_visibilities(Config *config, Visibility **visibilities, Complex **vis_intensity);

void create_perfect_image(Config *config, Complex *grid, Visibility *visibilities, Complex *vis_intensity);

void save_grid_to_file(Config *config, Complex *grid);

__global__ void inverse_dft_with_w_correction(double2 *grid, size_t grid_pitch, const double3 *visibilities, const double2 *vis_intensity,
		int vis_count, int batch_count, int x_offset, int y_offset, int render_size, double cell_size);

__device__ double2 complex_multiply(double2 z1, double2 z2);

float time_difference_msec(struct timeval t0, struct timeval t1);

void unit_test_init_config(Config *config);

double unit_test_generate_approximate_image(void);

#endif /* INVERSE_DIRECT_FOURIER_TRANSFORM_H_ */

#ifdef __cplusplus
}
#endif

