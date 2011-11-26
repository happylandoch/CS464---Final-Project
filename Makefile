
LIBS = -lGL -lglut -lGLEW

volumeRender: main.cpp volumeRender_kernel.cu
	nvcc -o $@ $(LIBS) $^

