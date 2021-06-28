//============================================================================
// Name        : FIlter.h
// Author      : Daniel Palomino
// Version     : 1.0
// Copyright   : GNU General Public License v3.0
// Description : Parallel Matrix Convolution Class
// Created on  : 06 set. 2018
//============================================================================

#ifndef FILTER_H
#define FILTER_H

#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include "Constants.h"


extern "C" void setConvolutionKernel(double* h_Kernel);
extern "C" void setConvolutionKernel2(double h_Kernel[KERNEL_LENGTH*KERNEL_LENGTH]);
extern "C" void convolutionGPU(
		double* image,
		double* result,
		int height,
		int width,
		int step);
extern "C" void growthMatrixGPU(double* matrix, double* result, int height, int width, int k);

#define ASSERT assert
//#define length(x) (sizeof(x)/sizeof(x[0]))

class Filter {

private:
	//double** dev_kernel;

public:
	Filter();
	Filter(double kernel[5*5]);
	Filter(double* kernel, int n);
	~Filter();

	bool setKernel(double kernel[5*5]);
	bool setKernel(double* kernel, int n);
	//bool showKernel();
	bool convolution(double* &image, double* &result, int x_length, int y_length, int step);

//Static methods
public:

	static bool showData(double** result, int x, int y);
	static bool showData(double* result, int x, int y);
	static bool showData(double result[5][5], int x, int y);
	static bool generateData(double** &matrix, int x, int y);
	static bool generateData(double* &matrix, int x, int y);

	static bool deleteMemory(double** &matrix, int x, int y);
	static bool reserveMemory(double** &matrix, int x, int y);
	static bool deleteMemory(double* &matrix, int x, int y);
	static bool reserveMemory(double* &matrix, int x, int y);

	static bool growthMatrix(double* matrix, double* result, int height, int width, int k);

};




#endif
