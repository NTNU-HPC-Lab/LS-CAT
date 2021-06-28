#pragma once
#include <cstdlib>

inline int div_ceil(int numerator, int denominator)
{
	std::div_t res = std::div(numerator, denominator);
	return res.quot + (res.rem != 0);
}


/* Multi threaded sort function. Executing 256 total steps (even and uneven) per kernel launch. */
void oddEvenSortCuda(int* arr, unsigned int arr_len);

/* Simple multi threaded sort function. Applying a single even or uneven step per thread and launch. */
void oddEvenSortCudaSimple(int* arr, unsigned int arr_len);

/* Simple multi threaded function executing multiple elements per thread on every even/uneven launch.*/
void oddEvenSortCudaMulti(int* arr, unsigned int arr_len, int threads);
