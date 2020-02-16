# Ray Tracing in One Weekend (RTOW) in Cuda with a GLFW live render

**DISCLAIMER: This code is intended to be close to the original RTOW tutorial and thus uses neither modern C++ nor is it pretty, production-ready or anything like that.**

![1800x900 Render of the current default scene at 3000 samples.](https://github.com/gomezzz/RTOWCudaGLFW/blob/master/example.png?raw=true)

The purpose of this code is to have a functional version of the RTOW code in CUDA with a live render output using GLFW. Thus, one can continue integration of more sophisticated features etc. without having to look at the generated image files all the time. If you have any ideas for improvements feel free to submit pull requests or comment.

I intentionally included my VS 2019 project files to allow a rapid start. Also, I added references in the code where each excerpt is discussed in RTOW and a bunch of comments.

The code builds on three different tutorials:

* [The original RTOW by Peter Shirley](https://raytracing.github.io/books/RayTracingInOneWeekend.html), an excellent introduction to ray tracing. [[Code]](https://github.com/RayTracing/raytracing.github.io/)
* [A recent adaptation of RTOW to CUDA by Roger Allen](https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/) that demonstrates a very straight forward implementation of the original RTOW code using CUDA. [[Code]](https://github.com/rogerallen/raytracinginoneweekendincuda)
* [A tiny example of CUDA + OpenGL interop by Allan MacKinnon](https://gist.github.com/allanmac/4ff11985c3562830989f) which was used here to integrate the GLFW live render display

## Structure

* *camera.h,hitable.h,hitable_list.h,material.h,ray.h,sphere.h,vec3.h* - These files are similar to the [RTOWCuda tutorial by Roger Allen](https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/)
* *check_cuda.h* - Used throughout the project to display potential CUDA errors
* *interop.cpp,interop.h* - These are close to the [tutorial by Allan MacKinnon](https://gist.github.com/allanmac/4ff11985c3562830989f)
* *main.cpp, kernel.cu* - These were modified to allow the live render display

For changing resolution, CUDA blocks / threads and the world (spheres etc.) configuration have a look at the settings section in *kernel.cu*

## Requirements
* [CUDA>=10.2](https://developer.nvidia.com/cuda-downloads) (lower might work, not tested)
* [GLFW](https://www.glfw.org/) - This will have to be added to includes and you need to link the appropriate lib
* [glad](https://glad.dav1d.de/) - You will need to include your own glad.c file into the project and add the includes
* [Visual Studio 2019](https://visualstudio.microsoft.com/vs/) - If you want to use the included VS solution file. Unfortunately, I don't have the time to create a cross-platform version using CMake right now. However, I am unaware of anything that would inhibit it.

For a great setup tutorial of GLFW and glad please refer to [learnopengl.com](https://learnopengl.com/Getting-started/Creating-a-window)

## Execution
Once you get the solution up and running you are good to go. Note that Release mode in VS will give you much higher FPS. Still, this code is of course not particularly efficient at the moment.

I took the freedom to change the default scene for testing purposes. If everything works you should get a lower resolution version of the image at the top.

