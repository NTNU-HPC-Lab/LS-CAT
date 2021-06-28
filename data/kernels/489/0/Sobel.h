/***
 * File: Sobel.h
 * Desc: Functions to apply a Sobel operator to a stencil.
 */

#ifndef SOBEL_H
#define SOBEL_H

#include "Stencil.h"

/*
 * Calculates the magnitude of the gradient of the stencil of image values
 * @param stencil -- the 3x3 image stencil
 * @return -- magnitude of the gradient
 */
double Sobel_Magnitude(Stencil_t *stencil);

#endif
