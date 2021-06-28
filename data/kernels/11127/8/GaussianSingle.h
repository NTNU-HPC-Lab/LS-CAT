#pragma once
#include <memory>

/* Row major matrix
*/
struct Matrix
{
	std::shared_ptr<float> arr;
	/* */
	int col, row;

	float* getRow(int row) { return &arr.get()[row*col]; }

	Matrix()
		: arr(), col(0), row(0)
	{}
	Matrix(int columns, int rows)
		: arr(new float[columns * rows]), col(columns), row(rows)
	{ }
	float& operator[](int index) { return arr.get()[index]; }
};

struct Vector
{
	std::shared_ptr<float> arr;
	int length;

	Vector()
		: arr(), length(0)
	{}
	Vector(int len)
		: arr(new float[len]), length(len)
	{}

	float& operator[](int index) { return arr.get()[index]; }
};


Vector gaussSolve(Matrix mat, Vector v);
Vector backSubstitute(Matrix mat, Vector b, int n);


void print(Matrix mat);
void print(Vector vec);
void print(Vector vec, int n);