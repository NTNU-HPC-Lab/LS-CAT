#ifndef SIMPLE_HPP
#define SIMPLE_HPP

#include<stdio.h>

void vector_add(float *out, float *a, float *b, int n);
void vector_print(float *in, int n);

/* The function to be wrapped by Cython */
void addition(float *out, float *a, float *b, int N);

#endif //SIMPLE_HPP
