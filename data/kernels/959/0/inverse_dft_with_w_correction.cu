#include "includes.h"

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




/**
* Check the return value of the CUDA runtime API call and exit
* the application if the call has failed.
*/
__device__ double2 complex_multiply(double2 z1, double2 z2)
{
double real = z1.x*z2.x - z1.y*z2.y;
double imag = z1.y*z2.x + z1.x*z2.y;
return make_double2(real, imag);
}
__global__ void inverse_dft_with_w_correction(double2 *grid, size_t grid_pitch, const double3 *visibilities, const double2 *vis_intensity, int vis_count, int batch_count, int x_offset, int y_offset, int render_size, double cell_size)
{
// look up id of thread
int idx = blockIdx.x*blockDim.x + threadIdx.x;
int idy = blockIdx.y*blockDim.y + threadIdx.y;

if(idx >= render_size || idy >= render_size)
return;

double real_sum = 0;
double imag_sum = 0;

// convert to x and y image coordinates
double x = (idx+x_offset) * cell_size;
double y = (idy+y_offset) * cell_size;

double2 vis;
double2 theta_complex = make_double2(0.0, 0.0);

// precalculate image correction and wCorrection
double image_correction = sqrt(1.0 - (x * x) - (y * y));
double w_correction = image_correction - 1.0;

// NOTE: below is an approximation... Uncomment if needed
// double wCorrection = -((x*x)+(y*y))/2.0;

// loop through all visibilities and create sum using iDFT formula
for(int i = 0; i < batch_count; ++i)
{
double theta = 2.0 * M_PI * (x * visibilities[i].x + y * visibilities[i].y
+ (w_correction * visibilities[i].z));
sincos(theta, &(theta_complex.y), &(theta_complex.x));
vis = complex_multiply(vis_intensity[i], theta_complex);
real_sum += vis.x;
imag_sum += vis.y;
}

// adjust sum by image correction
real_sum *= image_correction;
imag_sum *= image_correction;

// look up destination in image (grid) and divide by amount of visibilities (N)
double2 *row = (double2*)((char*)grid + idy * grid_pitch);
row[idx].x += (real_sum / vis_count);
row[idx].y += (imag_sum / vis_count);
}