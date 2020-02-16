#include <time.h>
#include <iostream>
#include "check_cuda.h"
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "device_launch_parameters.h"

surface<void, cudaSurfaceType2D> surf;

////////////////////////////////////////////
////////  Settings
////////////////////////////////////////////

const int resolution_downscale = 2; //use this to adjust resolution
extern "C" const int nx = 1800 / resolution_downscale;
extern "C" const int ny = 900 / resolution_downscale;
const int num_pixels = nx * ny;
const int N_samples = 100;

//One of these two should be removable, but I have had problems doing so
//First one used in world creation
const int tx = 8;
const int ty = 8;
const dim3 blocks(nx / tx + 1, ny / ty + 1);
const dim3 threads(tx, ty);

//This one is used for the main kernel
const int threads_1d = 512;
const int blocks_1d = (nx * ny + threads_1d - 1) / threads_1d;

////////////////////////////////////////////
////////  World, Camera, Randomstate
////////////////////////////////////////////
curandState* d_rand_state;
hitable** d_list;
hitable** d_world;
camera** d_camera;


//used for image conversion in the end
union rgbx_24
{
	uint1       b32;

	struct {
		unsigned  r : 8;
		unsigned  g : 8;
		unsigned  b : 8;
		unsigned  na : 8;
	};
};

//Given a ray, find out if it hit's anything in the world. If so, scatter it and continue, else just continue
__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0.0, 0.0, 0.0);
			}
		}
		else {
			//This code results in the blue sky background look
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		//Big blue sphere
		d_list[0] = new sphere(vec3(-3, 0.925, -3), 1.5, new lambertian(vec3(0.05, 0.05, 0.25)));
		//Giant black sphere below
		d_list[1] = new sphere(vec3(0, -100.5, -1), 100, new metal(vec3(0.025, 0.025, 0.025), 0.01));
		// Red sphere
		d_list[2] = new sphere(vec3(0, 0.4, -4), 1, new metal(vec3(0.5, 0.1, 0.1), 0.0));
		// Glass sphere
		d_list[3] = new sphere(vec3(0, 0, 0), 0.5, new dielectric(1.5));
		d_list[4] = new sphere(vec3(0, 0, 0), -0.45, new dielectric(1.5));

		//Create World from spheres
		*d_world = new hitable_list(d_list, 5);

		//Create Camera
		vec3 lookfrom(4, 1, 7);
		vec3 lookat(0, 0.5, -1);
		float dist_to_focus = (lookfrom - lookat).length();
		float aperture = 0.01;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			20.0,
			float(nx) / float(ny),
			aperture,
			dist_to_focus);
	}
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
	for (int i = 0; i < 4; i++) {
		delete ((sphere*)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete* d_world;
	delete* d_camera;
}

//Initializes the curand state 
__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curand_init(42 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

//Alloc and create everything in CUDA. Note that this function is called from the main to initiate setup (also why we need the extern "C")
extern "C" void setup_kernel()
{
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

	// make our world of hitables
	checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hitable*)));
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

	create_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// init RNG
	render_init << <blocks, threads >> > (nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

//Cleanup everything in CUDA. Note that this function is called from the main to initiate cleanup (also why we need the extern "C")
extern "C" void cleanup_kernel() {
	checkCudaErrors(cudaDeviceSynchronize());
	free_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
}

//Main render kernel
void __global__ kernel(int max_x, int max_y, int ns, camera * *cam, hitable * *world, curandState * rand_state)
{
	// pixel coordinates
	const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	const int x = idx % max_x;
	const int y = idx / max_x;

	//get indices
	if ((x >= max_x) || (y >= max_y)) return;
	int pixel_index = y * max_x + x;

	//get local random state for sampling
	curandState local_rand_state = rand_state[pixel_index];
	
	//color we'll determine
	vec3 col(0, 0, 0);

	//Run ns samples (more samples -> less noise)
	for (int s = 0; s < ns; s++) {
		float u = float(x + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(y + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);
	}

	rand_state[pixel_index] = local_rand_state; //remember random state changes
	col /= float(ns); //average accumulated colors

	//Write to the cuda surface which is bound to GLFW
	union rgbx_24  rgbx;
	rgbx.r = int(255.99 * sqrt(col[0]));
	rgbx.g = int(255.99 * sqrt(col[1]));
	rgbx.b = int(255.99 * sqrt(col[2]));
	rgbx.na = 255;

	surf2Dwrite(rgbx.b32,
		surf,
		x * sizeof(rgbx),
		max_y - y, //flip y
		cudaBoundaryModeZero); //squelches out-of-bound writes
}

extern "C" void execute_main_kernel(cudaArray_const_t array,
	cudaEvent_t       event,
	cudaStream_t      stream)
{
	checkCudaErrors(cudaBindSurfaceToArray(surf, array));

	kernel << <blocks_1d, threads_1d, 0, stream >> > (nx, ny, N_samples, d_camera, d_world, d_rand_state);

	checkCudaErrors(cudaDeviceSynchronize());
}
