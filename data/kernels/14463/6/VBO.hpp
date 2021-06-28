/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Warpper class for OpenGL vertex buffer objects (VBO) with support for CUDA
 *  interoperability
 */

#ifndef VBO_HPP
#define VBO_HPP

// System includes
#include "GLerror.hpp"

#ifndef QLB_HAS_CUDA

class VBO
{
public:

	/**
	 *	Constructor
	 *	@param	target	Specifies the target buffer object. 
	 *	                The symbolic constant must be one of : 
	 *	                GL_ARRAY_BUFFER 
	 *	                GL_COPY_READ_BUFFER 
	 *	                GL_COPY_WRITE_BUFFER 
	 *	                GL_ELEMENT_ARRAY_BUFFER 
	 *	                GL_PIXEL_PACK_BUFFER 
	 *	                GL_PIXEL_UNPACK_BUFFER 
	 *	                GL_TEXTURE_BUFFER 
	 *	                GL_TRANSFORM_FEEDBACK_BUFFER 
	 *	                GL_UNIFORM_BUFFER
	 *	@param 	usage	Specifies the expected usage pattern of the data store. 
	 *	                The symbolic constant must be one of :
	 *	                GL_STREAM_DRAW 
	 *	                GL_STREAM_READ 
	 *	                GL_STREAM_COPY 
	 *	                GL_STATIC_DRAW 
	 *	                GL_STATIC_READ 
	 *	                GL_STATIC_COPY 
	 *	                GL_DYNAMIC_DRAW
	 *	                GL_DYNAMIC_READ
	 *	                GL_DYNAMIC_COPY
	 */
	VBO(GLenum target, GLenum usage)
		: target_(target), usage_(usage), is_valid_(false)
	{
		init(target, usage);
	}

	/**
	 *	Constructor - default
	 */
	VBO()
		: id_(0), target_(0), usage_(0), is_valid_(false)
	{}

	/**
	 *	Destructor
	 */
	~VBO()
	{
		if(is_valid_)
			glDeleteBuffers(1, &id_);
	}

	/**
	 *	Initialize the buffer
	 *	@param target	see Constructor
	 *	@param usage	see Constructor
	 */
	inline void init(GLenum target, GLenum usage)
	{
		target_ = target;
		usage_ = usage;
		glGenBuffers(1, &id_);
		is_valid_ = true;
	}

	/** 
	 *	Creates and initializes the buffer (VBO must be bound)
	 *	@param	size	the size in bytes of the buffer object's 
	 *	                new data store.
	 *	@param 	data	a pointer to data that will be copied into the data 
	 *	                store for initialization
	 */
	inline void BufferData(GLsizeiptr size, const GLvoid* data)
	{
		glBufferData(target_, size, data, usage_);
	}

	/**
	 *	Allocate memory (VBO must be bound)
	 *	@param	size	the size in bytes of the buffer object's 
	 *	                new data store.
	 */
	inline void malloc(GLsizeiptr size)
	{
		glBufferData(target_, size, NULL, usage_); 
	}

	/**
	 *	Add data to the buffer in [begin, end] (VBO must be bound)
	 *	@param	begin	begin index of the data (in bytes)
	 *	@param	end		end index of the data (in bytes)
	 *	@param	data	a pointer to data that will be copied into the data 
	 *	                store for initialization
	 */
	inline void BufferSubData(GLsizeiptr begin, GLsizeiptr end, 
		                      const GLvoid* data)
	{
		glBufferSubData(target_, begin, end, data);
	}

	/**
	 *	Bind the VBO
	 */
	inline void bind()
	{
	    glBindBuffer(target_, id_); 
	}

	/**
	 *	Unbind the VBO
	 */
	inline void unbind()
	{
		glBindBuffer(target_, 0);
	}

	// === Getter ===
	inline GLuint id() const { return id_; }
	inline GLenum target() const { return target_; }
	inline GLenum usage() const { return usage_; }

private:
	GLuint id_; 
	GLenum target_;
	GLenum usage_;

	bool is_valid_;
};

#else 

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "cuassert.hpp"

class VBO
{
public:

	/**
	 *	Constructor
	 *	@param	target	Specifies the target buffer object. 
	 *	                The symbolic constant must be one of : 
	 *	                GL_ARRAY_BUFFER 
	 *	                GL_COPY_READ_BUFFER 
	 *	                GL_COPY_WRITE_BUFFER 
	 *	                GL_ELEMENT_ARRAY_BUFFER 
	 *	                GL_PIXEL_PACK_BUFFER 
	 *	                GL_PIXEL_UNPACK_BUFFER 
	 *	                GL_TEXTURE_BUFFER 
	 *	                GL_TRANSFORM_FEEDBACK_BUFFER 
	 *	                GL_UNIFORM_BUFFER
	 *	@param 	usage	Specifies the expected usage pattern of the data store. 
	 *	                The symbolic constant must be one of :
	 *	                GL_STREAM_DRAW 
	 *	                GL_STREAM_READ 
	 *	                GL_STREAM_COPY 
	 *	                GL_STATIC_DRAW 
	 *	                GL_STATIC_READ 
	 *	                GL_STATIC_COPY 
	 *	                GL_DYNAMIC_DRAW
	 *	                GL_DYNAMIC_READ
	 *	                GL_DYNAMIC_COPY
	 */
	VBO(GLenum target, GLenum usage)
		: target_(target), usage_(usage), is_valid_(false)
	{
		init(target, usage);
	}

	/**
	 *	Constructor - default
	 */
	VBO()
		: id_(0), target_(0), usage_(0), is_valid_(false)
	{}

	/**
	 *	Destructor
	 */
	~VBO()
	{
		if(is_valid_)
		{
			glDeleteBuffers(1, &id_);
			
			if(id_cuda_ != nullptr)
				cuassert(cudaGraphicsUnregisterResource(id_cuda_))
		}
	}	

	/**
	 *	Initialize the buffer
	 *	@param target	see Constructor
	 *	@param usage	see Constructor
	 */
	inline void init(GLenum target, GLenum usage)
	{
		target_ = target;
		usage_ = usage;
		glGenBuffers(1, &id_);
		is_valid_ = true;
	}

	/** 
	 *	Creates and initializes the buffer (VBO must be bound)
	 *	@param	size	the size in bytes of the buffer object's 
	 *	                new data store.
	 *	@param 	data	a pointer to data that will be copied into the data 
	 *	                store for initialization
	 */
	inline void BufferData(GLsizeiptr size, const GLvoid* data)
	{
		glBufferData(target_, size, data, usage_);
		
		unbind();
		cuassert(cudaGraphicsGLRegisterBuffer(&id_cuda_, id_, 
		                                      cudaGraphicsMapFlagsWriteDiscard));
		bind();
		
	}

	/**

	 *	Allocate memory (VBO must be bound)
	 *	@param	size	the size in bytes of the buffer object's 
	 *	                new data store.
	 */
	inline void malloc(GLsizeiptr size)
	{
		glBufferData(target_, size, NULL, usage_); 
		
		unbind();
		cuassert(cudaGraphicsGLRegisterBuffer(&id_cuda_, id_, 
		                                      cudaGraphicsMapFlagsWriteDiscard));
		bind();
	}

	/**
	 *	Add data to the buffer in [begin, end] (VBO must be bound)
	 *	@param	begin	begin index of the data (in bytes)
	 *	@param	end		end index of the data (in bytes)
	 *	@param	data	a pointer to data that will be copied into the data 
	 *	                store for initialization
	 */
	inline void BufferSubData(GLsizeiptr begin, GLsizeiptr end, 
		                      const GLvoid* data)
	{
		glBufferSubData(target_, begin, end, data);
	}

	/**
	 *	Bind the VBO
	 */
	inline void bind()
	{
	    glBindBuffer(target_, id_); 
	}

	/**
	 *	Unbind the VBO
	 */
	inline void unbind()
	{
		glBindBuffer(target_, 0);
	}
	
	/**
	 *	Map the VBO for CUDA
	 */
	inline void map()
	{
		cuassert(cudaGraphicsMapResources(1, &id_cuda_, 0));	
	}
	
	/**
	 *	Unmap the VBO for CUDA
	 */
	inline void unmap()
	{
		cuassert(cudaGraphicsUnmapResources(1, &id_cuda_, 0));	
	}
	
	/**
	 *	Get a device pointer to the VBO which can be used to modify the VBO by 
	 *	CUDA kernels (VBO must be mapped)
	 *	@return device pointer of type float3*
	 */
	inline float3* get_device_pointer()
	{
		float3 *dptr;
		std::size_t num_bytes;
		cuassert(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, 
		                                              id_cuda_));
		return dptr;	    
	} 	

	// === Getter ===
	inline GLuint id() const { return id_; }
	inline GLenum target() const { return target_; }
	inline GLenum usage() const { return usage_; }

private:
	GLuint id_; 
	GLenum target_;	
	GLenum usage_;
	cudaGraphicsResource_t id_cuda_;

	bool is_valid_;
};

#endif /* QLB_HAS_CUDA */

#endif /* VBO.hpp */
