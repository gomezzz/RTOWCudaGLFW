//Adapted from https://gist.github.com/allanmac/4ff11985c3562830989f by allanmac

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include "check_cuda.h"
#include "interop.h"

//Used to tell GLFW about the window size
extern "C" const int nx;
extern "C" const int ny;


//These functions are defined in the cuda code and called here during execution
extern "C" void execute_main_kernel(cudaArray_const_t array,
	cudaEvent_t       event,
	cudaStream_t      stream);

extern "C" void setup_kernel();

extern "C" void cleanup_kernel();

// FPS COUNTER FROM HERE:
// http://antongerdelan.net/opengl/glcontext2.html
//

static void glfw_fps(GLFWwindow* window)
{
	// static fps counters
	static double stamp_prev = 0.0;
	static int    frame_count = 0;

	// locals
	const double stamp_curr = glfwGetTime();
	const double elapsed = stamp_curr - stamp_prev;

	if (elapsed > 0.5)
	{
		stamp_prev = stamp_curr;

		const double fps = (double)frame_count / elapsed;

		int  width, height;
		char tmp[64];

		glfwGetFramebufferSize(window, &width, &height);

		sprintf_s(tmp, 64, "(%u x %u) - FPS: %.2f", width, height, fps);

		glfwSetWindowTitle(window, tmp);

		frame_count = 0;
	}

	frame_count++;
}

static void glfw_error_callback(int error, const char* description)
{
	fputs(description, stderr);
}


//Used to close window on ESC press
static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}

//Initializes GLFW
static void glfw_init(GLFWwindow** window, const int width, const int height)
{
	glfwSetErrorCallback(glfw_error_callback);

	if (!glfwInit())
		exit(EXIT_FAILURE);

	glfwWindowHint(GLFW_DEPTH_BITS, 0);
	glfwWindowHint(GLFW_STENCIL_BITS, 0);

	glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	*window = glfwCreateWindow(width, height, "GLFW / CUDA Interop", NULL, NULL);

	if (*window == NULL)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(*window);

	// set up GLAD
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

	// ignore vsync for now
	glfwSwapInterval(0);

	// only copy r/g/b
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);

	// enable SRGB 
	// glEnable(GL_FRAMEBUFFER_SRGB);
}

static void glfw_window_size_callback(GLFWwindow* window, int width, int height)
{
	// get context
	struct cuda_glfw_interop* const interop = (cuda_glfw_interop*)glfwGetWindowUserPointer(window);

	cuda_glfw_interop_size_set(interop, width, height);
}

int main(int argc, char* argv[])
{
	//Init GLFW
	GLFWwindow* window;
	glfw_init(&window, nx, ny);

	int gl_device_id;
	unsigned int gl_device_count;
	checkCudaErrors(cudaGLGetDevices(&gl_device_count, &gl_device_id, 1, cudaGLDeviceListAll));

	int cuda_device_id = (argc > 1) ? atoi(argv[1]) : gl_device_id;
	checkCudaErrors(cudaSetDevice(cuda_device_id));

	//Get device info
	struct cudaDeviceProp props;

	checkCudaErrors(cudaGetDeviceProperties(&props, gl_device_id));
	printf("GL   : %-24s (%2d)\n", props.name, props.multiProcessorCount);
	checkCudaErrors(cudaGetDeviceProperties(&props, cuda_device_id));
	printf("CUDA : %-24s (%2d)\n", props.name, props.multiProcessorCount);

	// CREATE CUDA STREAM & EVENT
	cudaStream_t stream;
	cudaEvent_t  event;

	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));   // optionally ignore default stream behavior
	checkCudaErrors(cudaEventCreateWithFlags(&event, cudaEventBlockingSync)); // | cudaEventDisableTiming);

	// CREATE INTEROP
	// TESTING -- DO NOT SET TO FALSE, ONLY TRUE IS RELIABLE
	struct cuda_glfw_interop* const interop = cuda_glfw_interop_create(true, 2);

	// RESIZE INTEROP
	int width, height;

	// get initial width/height
	glfwGetFramebufferSize(window, &width, &height);

	// resize with initial window dimensions
	cuda_glfw_interop_size_set(interop, width, height);

	// SET USER POINTER AND CALLBACKS
	glfwSetWindowUserPointer(window, interop);
	glfwSetKeyCallback(window, glfw_key_callback);
	glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);

	//INIT CUDA STUFF
	setup_kernel();

	// LOOP UNTIL DONE
	while (!glfwWindowShouldClose(window))
	{
		// MONITOR FPS
		glfw_fps(window);

		// EXECUTE CUDA KERNEL ON RENDER BUFFER
		int width, height;
		cuda_glfw_interop_size_get(interop, &width, &height);

		// Launch main kernel
		execute_main_kernel(cuda_glfw_interop_array_get(interop),event,stream);

		// BLIT & SWAP FBO
		cuda_glfw_interop_blit(interop);
		// cuda_glfw_interop_clear(interop);
		cuda_glfw_interop_swap(interop);

		// SWAP WINDOW
		glfwSwapBuffers(window);

		// PUMP/POLL/WAIT
		glfwPollEvents(); // glfwWaitEvents();
	}

	// CLEANUP
	cuda_glfw_interop_destroy(interop);
	glfwDestroyWindow(window);
	glfwTerminate();
	cleanup_kernel();
	checkCudaErrors(cudaDeviceReset());
	exit(EXIT_SUCCESS);
}