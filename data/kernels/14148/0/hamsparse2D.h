/* Copyright (C) 2012  Ward Poelmans

This file is part of Hubbard-GPU.

Hubbard-GPU is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Hubbard-GPU is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Hubbard-GPU.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef HAMSPARSE2D_H
#define HAMSPARSE2D_H

#include "hamsparse2D_CSR.h"

/**
 * Store the 2D Hubbard Hamiltonian in the ELL format
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
class SparseHamiltonian2D: public SparseHamiltonian2D_CSR
{
    public:
	SparseHamiltonian2D(int L, int D, int Nu, int Nd, double J, double U);
	virtual ~SparseHamiltonian2D();

	void BuildSparseHam();

	void PrintSparse() const;

	void PrintRawELL() const;

        virtual void mvprod(double *, double *, double) const;

    protected:

	//! The array to hold the data (ELL format) for the up hamiltonian
	double *Up_data;
	//! The array to hold the data (ELL format) for the down hamiltonian
	double *Down_data;

	//! The array to hold the indices for the up hamiltonian
	unsigned int *Up_ind;
	//! The array to hold the indices for the down hamiltonian
	unsigned int *Down_ind;

	//! Maximum number of non zero elements in a row for the up hamiltonian
	int size_Up;
	//! Maximum number of non zero elements in a row for the down hamiltonian
	int size_Down;

};

#endif /* HAMSPARSE2D_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
