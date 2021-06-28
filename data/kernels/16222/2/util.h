#include <iostream>
#include <fstream>


int * read_file(char * name, int * row, int * col) {
    std::ifstream infile(name);

    infile >> *col >> *row;

    int * data = (int *)malloc(sizeof(int) * *row * *col);

    for (int i = 0; i < *row * *col; i++) {
        infile >> data[i];
    }

    infile.close();

    return data;
}