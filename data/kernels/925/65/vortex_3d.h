/*-------------vortex_3d.cu---------------------------------------------------//
*
* Purpose: This file intends to perform a convolution of 3d data for vortex
*          recognition in the GPUE code
*
*   Notes: We will be using the window method for convolutions because it has
*          a slightly better complxity case for a separable filter
*          (which the Sobel filter definitely is!)
*
*-----------------------------------------------------------------------------*/

#ifndef VORTEX_3D_H
#define VORTEX_3D_H

//#include "kernels.h"
#include <stdio.h>

// Kernel to return vortex positions

// Kernel to return spine of edges
pos **find_skeletons(double *edges);

// Function to find 3d sobel operators for fft convolution
void find_sobel(Grid &par);
void find_sobel_2d(Grid &par);

// Function to transfer 3d sobel operators for non-fft convolution
void transfer_sobel(Grid &par);

// We need a central kernel with just inputs and outputs
void find_edges(Grid &par,
                double2* wfc, double* edges);

#endif
