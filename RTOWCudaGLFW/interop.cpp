//Adapted from https://gist.github.com/allanmac/4ff11985c3562830989f by allanmac

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <stdlib.h>

#include "check_cuda.h"
#include "interop.h"


struct cuda_glfw_interop
{
	// split GPUs?
	bool multi_gpu;

	// number of fbo's
	int count;
	int index;

	// w x h
	int width;
	int height;

	// GL buffers
	GLuint* fb;
	GLuint* rb;

	// CUDA resources
	cudaGraphicsResource_t* cgr;
	cudaArray_t* ca;
};

//
//
//

struct cuda_glfw_interop* cuda_glfw_interop_create(const bool multi_gpu, const int fbo_count)
{
	struct cuda_glfw_interop* const interop = (cuda_glfw_interop*)calloc(1, sizeof(*interop));

	interop->multi_gpu = multi_gpu;
	interop->count = fbo_count;
	interop->index = 0;

	// allocate arrays
	interop->fb = (GLuint*)calloc(fbo_count, sizeof(*(interop->fb)));
	interop->rb = (GLuint*)calloc(fbo_count, sizeof(*(interop->rb)));
	interop->cgr = (cudaGraphicsResource_t*)calloc(fbo_count, sizeof(*(interop->cgr)));
	interop->ca = (cudaArray_t*)calloc(fbo_count, sizeof(*(interop->ca)));

	// render buffer object w/a color buffer
	glCreateRenderbuffers(fbo_count, interop->rb);

	// frame buffer object
	glCreateFramebuffers(fbo_count, interop->fb);

	// attach rbo to fbo
	for (int index = 0; index < fbo_count; index++)
	{
		glNamedFramebufferRenderbuffer(interop->fb[index],
			GL_COLOR_ATTACHMENT0,
			GL_RENDERBUFFER,
			interop->rb[index]);
	}

	// return it
	return interop;
}


void cuda_glfw_interop_destroy(struct cuda_glfw_interop* const interop)
{
	// unregister CUDA resources
	for (int index = 0; index < interop->count; index++)
	{
		if (interop->cgr[index] != NULL)
			checkCudaErrors(cudaGraphicsUnregisterResource(interop->cgr[index]));
	}

	// delete rbo's
	glDeleteRenderbuffers(interop->count, interop->rb);

	// delete fbo's
	glDeleteFramebuffers(interop->count, interop->fb);

	// free buffers and resources
	free(interop->fb);
	free(interop->rb);
	free(interop->cgr);
	free(interop->ca);

	// free interop
	free(interop);
}

void cuda_glfw_interop_size_set(struct cuda_glfw_interop* const interop, const int width, const int height)
{
	// save new size
	interop->width = width;
	interop->height = height;

	// resize color buffer
	for (int index = 0; index < interop->count; index++)
	{
		// unregister resource
		if (interop->cgr[index] != NULL)
			checkCudaErrors(cudaGraphicsUnregisterResource(interop->cgr[index]));

		// resize rbo
		glNamedRenderbufferStorage(interop->rb[index], GL_RGBA8, width, height);

		// probe fbo status
		// glCheckNamedFramebufferStatus(interop->fb[index],0);

		// register rbo
		checkCudaErrors(cudaGraphicsGLRegisterImage(&interop->cgr[index],
			interop->rb[index],
			GL_RENDERBUFFER,
			cudaGraphicsRegisterFlagsSurfaceLoadStore |
			cudaGraphicsRegisterFlagsWriteDiscard));
	}

	// map graphics resources
	checkCudaErrors(cudaGraphicsMapResources(interop->count, interop->cgr, 0));

	// get CUDA Array refernces
	for (int index = 0; index < interop->count; index++)
	{
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&interop->ca[index],
			interop->cgr[index],
			0, 0));
	}

	// unmap graphics resources
	checkCudaErrors(cudaGraphicsUnmapResources(interop->count, interop->cgr, 0));
}

void cuda_glfw_interop_size_get(struct cuda_glfw_interop* const interop, int* const width, int* const height)
{
	*width = interop->width;
	*height = interop->height;
}


void cuda_glfw_interop_array_map(struct cuda_glfw_interop* const interop)
{
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&interop->ca[interop->index],
		interop->cgr[interop->index],
		0, 0));
}

cudaArray_const_t cuda_glfw_interop_array_get(struct cuda_glfw_interop* const interop)
{
	return interop->ca[interop->index];
}

int cuda_glfw_interop_index_get(struct cuda_glfw_interop* const interop)
{
	return interop->index;
}

void cuda_glfw_interop_swap(struct cuda_glfw_interop* const interop)
{
	interop->index = (interop->index + 1) % interop->count;
}

void cuda_glfw_interop_clear(struct cuda_glfw_interop* const interop)
{
	/*
	static const GLenum attachments[] = { GL_COLOR_ATTACHMENT0 };
	glInvalidateNamedFramebufferData(interop->fb[interop->index],1,attachments);
	*/

	const GLfloat clear_color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glClearNamedFramebufferfv(interop->fb[interop->index], GL_COLOR, 0, clear_color);
}

void cuda_glfw_interop_blit(struct cuda_glfw_interop* const interop)
{
	glBlitNamedFramebuffer(interop->fb[interop->index], 0,
		0, 0, interop->width, interop->height,
		0, interop->height, interop->width, 0,
		GL_COLOR_BUFFER_BIT,
		GL_NEAREST);
}