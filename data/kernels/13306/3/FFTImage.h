/*================================================================
 * Display Images with CPU
 *----------------------------------------------------------------
 * Licence isn't exists.
 *
 * vmg.h
 *
 * Copyright (c) 2012 NULL
 *
 *================================================================*/

#pragma once

#include "Typedefs.h"

namespace Imgproc
{
	void	Fft1D(double *lpDstRe, double *lpDstIm, double *lpSrcRe, double *lpSrcIm, int log2n);
	void	Ifft1D(double *lpDstRe, double *lpDstIm, double *lpSrcRe, double *lpSrcIm, int log2n);
	void	Fft2D(double *lpDstRe, double *lpDstIm, double *lpSrcRe, double *lpSrcIm, int width, int height);
	void	IFft2D(double *lpDstRe, double *lpDstIm, double *lpSrcRe, double *lpSrcIm, int width, int height);
	void	FFTImage(UINT32 *lpDst, UINT32 *lpSrc, int width, int height);
	void	FFTPhaseImage(UINT32 *lpDst, UINT32 *lpSrc, int width, int height);

	void	DCuFFTImage(UINT32 *d_lpDst, UINT32 *d_lpSrc, float *d_lpTmp, int width, int height);
	void	DCuFFTPhaseImage(UINT32 *d_lpDst, UINT32 *d_lpSrc, int width, int height);
}
