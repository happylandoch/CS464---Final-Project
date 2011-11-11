## Add source files here
#EXECUTABLE	 := volumeRender
## Cuda source files (compiled with cudacc)
CUFILES		 := volumeRender_kernel.cu

# C/C++ source files (compiled with gcc / c++)
CCFILES		 := volumeRender.cpp

#USEGLLIB         := 1
#USEGLUT	         := 1
#USERENDERCHECKGL := 1


volumeRender: volumeRender.cpp volumeRender_kernel.cu
	nvcc -c volumeRender_kernel.cu

