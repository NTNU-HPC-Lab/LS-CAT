/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Providing OpenGL utility functions to handle runtime errors.
 *  This file will also include the OpenGL and GLUT headers.
 */

#ifndef GL_ERROR_HPP
#define GL_ERROR_HPP

// System includes
#include <iostream>
#include <string>

// OpenGL includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
 #include <GLUT/glut.h>
 #ifndef glutCloseFunc
 #define glutCloseFunc glutWMCloseFunc
 #endif
#else
 #include <GL/freeglut.h>
#endif

// Local includes
#include "error.hpp"


#define glCheckLastError() glCheckLastError_(__FILE__, __LINE__)
static inline void glCheckLastError_(const char* file, int line)
{
	GLenum error = glGetError();
	bool abort_after_error = false;
	std::string errmsg;

	while(error)
	{
		abort_after_error = true;
		switch(error)
		{
			case GL_INVALID_ENUM:
				errmsg += "GL_INVALID_ENUM ";
				break;
			case GL_INVALID_VALUE:
				errmsg += "GL_INVALID_VALUE ";
				break;
			case GL_INVALID_OPERATION:
				errmsg += "GL_INVALID_OPERATION ";
				break;
			case GL_INVALID_FRAMEBUFFER_OPERATION:
				errmsg += "GL_INVALID_FRAMEBUFFER_OPERATION ";
				break;
			case GL_OUT_OF_MEMORY:
				errmsg += "GL_OUT_OF_MEMORY ";
				break;
			case GL_STACK_UNDERFLOW:
				errmsg += "GL_STACK_UNDERFLOW ";
				break;
			case GL_STACK_OVERFLOW:
				errmsg += "GL_STACK_OVERFLOW ";
				break;
			default:
				errmsg += "unknown error code "; 
				break;
		}

		// Get the next error in the stack
		error = glGetError();
	}

	if(abort_after_error)
		fatal_error(errmsg, file, line);
}

#endif /* GLerror.hpp */
