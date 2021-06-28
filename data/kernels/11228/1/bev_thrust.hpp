#ifndef BEV_THRUST_HPP
#define BEV_THRUST_HPP

#include<iostream>
#include<cuda_runtime_api.h>
#include<cuda.h>
#include<vector>
#include<math.h>
#include<cstring>
#include<limits>
#include <iomanip>


/*Thrust Definitions*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;



#define TILE_WIDTH 2
#define IMAGE_WIDTH_GRAY 1242
#define IMAGE_HEIGHT_GRAY 375


typedef vector<float> row_t;
typedef vector<row_t> matrix_t;

typedef thrust::host_vector<float> h_row_t;
typedef thrust::host_vector<float> h_matrix_t;

typedef thrust::device_vector<float> d_row_t;
typedef thrust::device_vector<float> d_matrix_t;

typedef thrust::tuple<float, float> tuple_t;
typedef thrust::device_vector<float>::iterator floatIterator;
typedef thrust::tuple<floatIterator, floatIterator> floatIteratorTuple;
typedef thrust::zip_iterator<floatIteratorTuple> zipIterator;
typedef thrust::device_vector<tuple_t>::iterator tupleIterator;


typedef struct coord{

	int a;
	int b;
}tuple_int;

void printiarray(int* arr, int row, int column);
void printfarray(float* arr, int row, int column);
void print1dvector(row_t vector);
void print2dvector(matrix_t vector);

__global__ void matrix_mul(float* d_A, float* d_B, float* d_C, int numARows, int numAColumns, int numBRows, int numBColumns,
		int numCRows, int numCColumns);

float* getMatrix(matrix_t, float*, int, int);

/*Function to fo matrix_multiplication*/
matrix_t matrix_multiplication(matrix_t const& vec_a, matrix_t const& vec_b);

/*Functions to obtain inverse of a matrix*/
void getCofactor(matrix_t &vec_a, matrix_t &vec_b, int p, int q, int vec_a_rows);

double determinant(matrix_t &vec_a, int n);
matrix_t adjoint(matrix_t &vec_a);
matrix_t inverse(matrix_t &vec_a);


class BevParams
{   
	public:
		tuple_int bev_size;
		float bev_res;
		tuple_int bev_xLimits;
		tuple_int bev_zLimits;
		tuple_int imSize;
		/*Constructor*/ 
		BevParams(float bev_res, tuple_int bev_xLimits, tuple_int bev_zLimits, tuple_int imSize);

};


class Calibration
{
	public:
		matrix_t P2;
		matrix_t R0_Rect;
		matrix_t Tr_cam_to_road;
		matrix_t Tr33;
		matrix_t Tr33_inverse;
		matrix_t Tr;
		/*Constructor*/
		Calibration();
		void setup_calib(matrix_t P2, matrix_t R0_Rect, matrix_t Tr_cam_to_road);
		matrix_t get_matrix33();
		matrix_t get_matrix33_inverse();

};




class BirdsEyeView
{	  
	public:
		tuple_int imSize; /*RGB Image*/
		BevParams* bevParams;	
		Calibration* calib;
		float invalid_value;
		float bev_res;
		tuple_int bev_xRange_minMax;
		tuple_int bev_zRange_minMax;
		matrix_t Tr33;
		matrix_t Tr33_inverse;
		matrix_t uvMat;
		matrix_t uvMat_reverse;
		//float* h_B;
		float* h_B;
		static float* s_h_B;
		int numBRows;
		int numBColumns;
		void computeLookUpTable();	
		float* xi_1;
		float* yi_1;
		vector<int> z_index_vec;
		vector<int> x_index_vec;
		vector<int> x_bev_index_sel;
		vector<int> z_bev_index_sel;
		vector<int> x_im_index_sel;
		vector<int> y_im_index_sel;


		BirdsEyeView(float bev_res,double invalid_value, tuple_int bev_xRange_minMax, tuple_int bev_zRange_minMax);
		void setup(matrix_t P2, matrix_t R0_Rect, matrix_t Tr_cam_to_road);
		void set_matrix33(matrix_t Tr33);
		void set_matrix33_inverse(matrix_t Tr33_inverse);
		//void initialize(Mat& image);
		void initialize();
		static float* getWorld() {return s_h_B ;};
		//void computeLookUpTable(Mat& image);
		unsigned char* computeLookUpTable(unsigned char* image);
		unsigned char* getperspectiveView(unsigned char* image);
};

#endif
