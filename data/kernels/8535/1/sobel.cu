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


__global__ void sobel(unsigned char *src, unsigned char *dest, int width, int heigth, int step, int channels){
int i, j;
int ren, col, tmp_ren, tmp_col;
int gx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}}; // gx is defined in the Sobel algorithm
int gy[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}}; // gy is defined in the Sobel algorithm
char temp[3][3];

ren = blockIdx.x;
col = threadIdx.x;

tmp_ren = 0;
tmp_col = 0;

//Multiplication of the 3x3 matrix for each color
for (i = -1; i < 2; i++) {
for (j = -1; j < 2; j++) {
temp[i+1][j+1]=(int) src[(ren * step) + (col * channels) + RED + i + 1];
tmp_ren=tmp_ren + temp[i+1][j+1]*gx[i+1][j+1];
tmp_col=tmp_col + temp[i+1][j+1]*gy[i+1][j+1];
}
}
dest[(ren * step) + (col * channels) + RED] =  (unsigned char) sqrtf(tmp_col*tmp_col+tmp_ren*tmp_ren);;

tmp_ren = 0;
tmp_col = 0;
for (i = -1; i < 2; i++) {
for (j = -1; j < 2; j++) {
temp[i+1][j+1]=(int) src[(ren * step) + (col * channels) + GREEN + i + 1];
tmp_ren=tmp_ren + temp[i+1][j+1]*gx[i+1][j+1];
tmp_col=tmp_col + temp[i+1][j+1]*gy[i+1][j+1];
}
}
dest[(ren * step) + (col * channels) + GREEN] =  (unsigned char) sqrtf(tmp_col*tmp_col+tmp_ren*tmp_ren);;


tmp_ren = 0;
tmp_col = 0;
for (i = -1; i < 2; i++) {
for (j = -1; j < 2; j++) {
temp[i+1][j+1]=(int) src[(ren * step) + (col * channels) + BLUE + i + 1];
tmp_ren=tmp_ren + temp[i+1][j+1]*gx[i+1][j+1];
tmp_col=tmp_col + temp[i+1][j+1]*gy[i+1][j+1];
}
}
dest[(ren * step) + (col * channels) + BLUE] =  (unsigned char) sqrtf(tmp_col*tmp_col+tmp_ren*tmp_ren);
}