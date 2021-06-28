
#ifndef MUL_H
#define MUL_H

/* Multiply the nxn matrices a and b, placing the result in c. Matrices
 * are passed as 1D arrays of floats, because we don't know the dimensions
 * of the matrices at compile time. The last argument specifies the length
 * of one side of the matrix.
 */
void mul(float c[], float a[], float b[], int n);


#endif

