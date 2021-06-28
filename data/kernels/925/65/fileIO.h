///@endcond
//##############################################################################
/**
 *  @file    fileIO.h
 *  @author  Lee J. O'Riordan (mlxd)
 *  @date    12/11/2015
 *  @version 0.1
 *
 *  @brief Routines for input and output of simulation data.
 *
 *  @section DESCRIPTION
 *  The functions herein are used to write the simulation data to text-based
 *  files (HDF was planned, but for simplicity I removed it). Data from previous
 *  simulations can also be read into memory.
 */
 //##############################################################################

#ifndef FILEIO_H
#define FILEIO_H
#include "../include/ds.h"
#include "../include/tracker.h"
#include <vector>
#include <string>

/** Check source file for further information on functions **/
namespace FileIO {

    /**
    * @brief	Reads in the real and imaginary components from text files
    * @ingroup	helper
    *
    * @param	*fileR Name of data file of real components
    * @param	*fileI Name of data file of imaginary components
    * @param	xDim Size of x-grid
    * @param	yDim Size of y-grid
    * @return	*double2 Memory address of read-in data. Complex only
    */
    double2 *readIn(std::string fileR, std::string fileI, int gSize);

    /**
    * @brief	Writes the specified double2 array to a text file
    * @ingroup	helper
    *
    * @param	*buffer Char buffer for use by function internals. char[100] usually
    * @param	*file Name of data file name for saving to
    * @param	*data double2 array to be written out
    * @param	length Overall length of the file to write out
    * @param	step Index for the filename. file_step,filei_step
    */
    void writeOut(std::string buffer, std::string file, double2 *data, int length, int step);

	/**
    * @brief	Writes the specified double array to a text file
    * @ingroup	helper
    *
    * @param	*buffer Char buffer for use by function internals. char[100] usually
    * @param	*file Name of data file name for saving to
    * @param	*data double array to be written out
    * @param	length Overall length of the file to write out
    * @param	step Index for the filename. file_step
    */
    void writeOutDouble(std::string buffer, std::string file, double *data,
                        int length, int step);

        /**
    * @brief    Writes the specified double array to a text file
    * @ingroup  helper
    *
    * @param    *buffer Char buffer for use by function internals. char[100] usually
    * @param    *file Name of data file name for saving to
    * @param    *data bool array to be written out
    * @param    length Overall length of the file to write out
    * @param    step Index for the filename. file_step
    */
    void writeOutBool(std::string buffer, std::string file, bool *data,
                        int length, int step);


	/**
    * @brief	Writes the specified int array to a text file
    * @ingroup	helper
    *
    * @param	*buffer Char buffer for use by function internals. char[100] usually
    * @param	*file Name of data file name for saving to
    * @param	*data int array to be written out
    * @param	length Overall length of the file to write out
    * @param	step Index for the filename. file_step
    */
    void writeOutInt(std::string buffer, std::string file, int *data,
                     int length, int step);

	/**
    * @brief	Writes the specified int2 array to a text file
    * @ingroup	helper
    *
    * @param	*buffer Char buffer for use by function internals. char[100] usually
    * @param	*file Name of data file name for saving to
    * @param	*data int2 array to be written out
    * @param	length Overall length of the file to write out
    * @param	step Index for the filename. file_step
    */
    void writeOutInt2(std::string buffer, std::string file, int2 *data,
                      int length, int step);

	/**
    * @brief	Writes the specified Vtx::Vortex array to a text file
    * @ingroup	helper
    *
    * @param	*buffer Char buffer for use by function internals. char[100] usually
    * @param	*file Name of data file name for saving to
    * @param	*data Vtx::Vortex array to be written out
    * @param	step Index for the filename. file_step
    */
    void writeOutVortex(std::string buffer, std::string file,
                        std::vector<std::shared_ptr<Vtx::Vortex>> &data, int step);

	/**
    * @brief	Writes the parameter file
    * @ingroup	helper
    *
    * @param	*buffer Char buffer for use by function internals. char[100] usually
    * @param	arr struct Array holding the parameter values to be written out
    * @param	*file Name of data file name for saving to
    */
    void writeOutParam(std::string buffer, Grid &par, std::string file);

	/*
	 * @brief	Opens and closes file. Nothing more. Nothing less.
	 * @param	file Name of file to open
	 * @return	int 0. That's all.
	 */
    int readState(std::string name);

	/**
    * @brief	Write adjacency matrix of ints to a file in Mathematica readable format
    * @ingroup	graph
    *
    * @param	*buffer Char buffer for use by function internals. char[100] usually
    * @param	*file Name of data file name for saving to
	* @param	*mat Int Array holding the parameter values to be written out
	* @param	*uids UID array
	* @param	dim Dimension/length of the grid (xDim*yDim)
	* @param	step Index for the filename.
    */
    void writeOutAdjMat(std::string buffer, std::string file, int *mat, unsigned int *uids, int dim, int step);

	/**
    * @brief	Write adjacency matrix of doubles to a file in Mathematica readable format
    * @ingroup	graph
    *
    * @param	*buffer Char buffer for use by function internals. char[100] usually
    * @param	*file Name of data file name for saving to
	* @param	*mat double Array holding the parameter values to be written out
	* @param	*uids UID array
	* @param	dim Dimension/length of the grid (xDim*yDim)
	* @param	step Index for the filename.
    */
    void writeOutAdjMat(std::string buffer, std::string file, double *mat,
                        unsigned int *uids, int dim, int step);
}
#endif
