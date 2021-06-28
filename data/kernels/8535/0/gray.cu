#include "includes.h"
/* This code will generate a Sobel image and a Gray Scale image. Uses OpenCV, to compile:
nvcc FinalProject.cu `pkg-config --cflags --libs opencv`

Copyright (C) 2018  Jose Andres Cortez Villao

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.*/



typedef enum color {BLUE, GREEN, RED} Color;	//Constants that contains the values for each color of the image

/*The gray function obtain an average of each pixel and assigned to the correct position in the array using
Channels and step constants*/
/*The sobel function uses a convolution algorithm to obtain the edges of the image */


__global__ void gray(unsigned char *src, unsigned char *dest, int width, int heigth, int step, int channels) {
int ren, col;
float r, g, b;

ren = blockIdx.x; // Variables that parallelize the code
col = threadIdx.x;
r = 0; g = 0; b = 0;

r += (float) src[(ren * step) + (col * channels) + RED];
g += (float) src[(ren * step) + (col * channels) + GREEN];
b += (float) src[(ren * step) + (col * channels) + BLUE];

dest[(ren * step) + (col * channels) + RED] =  (unsigned char) ((r+g+b)/3);
dest[(ren * step) + (col * channels) + GREEN] = (unsigned char) ((r+g+b)/3);
dest[(ren * step) + (col * channels) + BLUE] = (unsigned char) ((r+g+b)/3);
}