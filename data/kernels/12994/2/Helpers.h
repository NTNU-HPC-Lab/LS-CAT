#ifndef _Helpers_h_
#define _Helpers_h_



int LoadGridFromFile(const int width, const int height, int* grid, char* filename);
void printCells(int* cells, int width, int height);
void randomMap(int *i_cells, int width, int height);
void usage(char* name);
#endif
