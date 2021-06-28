#pragma once
#include "GaussianSingle.h"



Vector gaussSolveCudaDevice(Matrix& mat, Vector& v);



//Some mem. test funcs:


void perWarpTransaction();
void singleWarpTransaction();