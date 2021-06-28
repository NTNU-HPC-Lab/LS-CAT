/*
 *  arrayUtils.h
 *  MakeMatrix
 *
 */

#ifndef ARRAYUTILS_H
#define ARRAYUTILS_H


/** Print a text representation of a 2D float array to stdout. 
 *
 *  @param arr a pointer to the 2D array
 *  @param rows the number of rows in the array
 *  @param cols the number of columns in the array
 */
void printArray(float *arr, int rows, int cols);

/** Read a 2D float array from a file. Returns a pointer to a newly allocated 
 *  array of floats containing the data read from the named file. The file
 *  must be text, and must begin with two ints specifying the number
 *  of rows and columns. These values are returned by the reference
 *  parameters. The remainder of the file should specify the contents
 *  of the array. 
 *
 *  On error, this function will return NULL. 
 *
 *  @param rows a pointer to an int that will receive the number of rows
 *  @param cols a pointer to an int that will receive the number of columns
 *  @param filename the name of the file to read from
 */
float * readNewArray(int *rows, int *cols, const char * filename);


/** Write a 2D float array to a file. The file will be text, and will begin 
 *  with two ints specifying the number of rows and columns. The remainder of
 *  the file will specify the contents of the array. 
 *
 *  @param arr a pointer to the 2D array
 *  @param rows the number of rows in the array
 *  @param cols the number of columns in the array
 *  @param filename the name of the file to write to.
 */
void writeArray(float *arr, int rows, int cols, const char * filename);



#endif

