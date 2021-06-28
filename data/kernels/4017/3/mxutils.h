#ifndef MXUTILS_H
#define MXUTILS_H
#include <cufft.h>
#include <complex.h>
#include <fftw3.h>
#include "easyloggingcpp/easylogging++.h"
__global__ void kronecker(double * a, int nax, int nay, double * b, int nbx, int nby, double * result);

void saveMatrixToCSV(std::string fname, double * mx, size_t szX, size_t szY);
void saveMatrixToCSV(std::string fnameRe, std::string fnameIm, fftw_complex * mx, size_t szX, size_t szY);

double * parseMxFromCSV(std::string fn, size_t &szX, size_t &szY);

int copyToCudaComplex(cufftDoubleComplex * d_complexData, size_t pitch, fftw_complex * h_complexWF, size_t szX, size_t szY);


cufftDoubleComplex * makeExpV(double dt, const double * d_Vt, size_t vSize, double g, double dx);

/*!
  Copy array from GPU to already allocated memory on the host
  Better used with small and 1D arrays and other auxiliary data,
  as it doesn't recognize of the data is pitched
  \param h_data : pointer to already allocated memory on the host
  \param d_data : pointer to the data to copy from on the GPU
  \param dataSz : size of the data in units of sizeof(T)
  \return 0 if success and -1 otherwise. Logs the error using easylogging
*/
template <typename T> int copyArrFromGPU(T * h_data, const T * d_data, size_t dataSz){
  if (h_data == NULL || dataSz == 0 || d_data == NULL)
  {
    LOG(ERROR) << __func__ << " : input data is invalid\n";
    return -1;
  }
  cudaError_t err = cudaMemcpy(h_data, d_data, sizeof(T) * dataSz, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << " : Cuda memory copy failed :" << cudaGetErrorString(err) << "\n";
    return -1;
  }
  return 0;
}

/*!
  Allocate memory on the GPU and copy a host array to that memory
  Is better used for small or 1D arrays and other auxiliary data,
  as it doesn't pitch the 2D arrays.
  NOTE: clean up with cudaFree
  \param h_data : pointer to host data
  \param dataSz : size of the dataSz
  \return pointer to device data of size dataSz
*/
template <typename T> T * allocateAndCopyToGPU(const T * h_data, size_t dataSz){
  if (h_data == NULL || dataSz == 0)
  {
    LOG(ERROR) << __func__ << " : input data is invalid\n";
    return NULL;
  }
  T * d_data = NULL;
  cudaError_t err = cudaMalloc(&d_data, sizeof(T) * dataSz);
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << " : Cuda memory allocation failed" << cudaGetErrorString(err) << "\n";
    return NULL;
  }
  err = cudaMemcpy(d_data, h_data, sizeof(T) * dataSz, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << " : Cuda memory copy failed" << cudaGetErrorString(err) << "\n";
    return NULL;
  }
  return d_data;
}

/*!
  Perform FFT on device data
  \param d_data : pointer to device data
  \param szX : size of the data's fastest changing dimension
  \param szY : size of the data's slower changing dimension
  \param plan : pointer to a cufft plan structure. If plan == NULL, new cufft plan is going to be created and its pointer saved in plan variable
  \param forward : foward FFT will be performed if true, inverse otherwise
  \return 0 if succes and -1  otherwise
*/
int doFFT(cufftDoubleComplex * d_data, size_t szX, size_t szY, cufftHandle * &plan, bool forward = true);

/*!
  Get the norm of the wave function |f| = sqrt(sum(|f_ij|^2)). Assume dx = 1;
  To get real norm, multiply the results by real dx
  \param d_wf : the sz x sz wave function allocated on the device
  \param sz : wave function size (dim(d_wf) = sz x sz)
  \param result : variable to save the result in
  \return 0 if success and -1 otherwise
*/
int getNorm(cufftDoubleComplex * d_wf, size_t sz, double & result);

/*!
  Normalize device wave function psi = psi/(dx * Sqrt(Sum(|psi[i]|^2 ))). I assume its a 2D wf
  \param dx : step for integration
  \param d_wf : pointer to the wave function in device memory
  \param sz : the dimensions if the 2D wf is sz x sz
  \return 0 if success and -1 otherwise
*/
int normalize(cufftDoubleComplex * d_wf, size_t sz, double dx);

/*!
Generate uniform random pick from (0, N-1) with replacement
Suppose you have a deck of 100 cards, with the numbers 1-100 on one side. You select a card, note the number, replace the card, shuffle, and repeat.
  \param d_dst : destination array to store the result on device.
  \param N: number of elements to pick
  \param maxThreads : max thread per block
  \return 0 if all goes well and -1 otherwise
*/
int generateRandomPicks(size_t * d_dst, size_t N, unsigned int maxThreads = 1024);

//! Fill the array on device with 0,...,N-1
int fillRange(size_t * d_dst, size_t N, unsigned int maxThreads);
//! Permute array randomly on the device
int permuteElements(size_t * d_dst, size_t N, unsigned int maxThreads);
//! Permute array randomly on the host
int permuteElements_host(size_t * h_dst, size_t N);
#endif
