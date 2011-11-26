/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Volume rendering sample

    This sample loads a 3D volume from disk and displays it using
    ray marching and 3D textures.

    Note - this is intended to be an example of using 3D textures
    in CUDA, not an optimized volume renderer.

    Changes
    sgg 22/3/2010
    - updated to use texture for display instead of glDrawPixels.
    - changed to render from front-to-back rather than back-to-front.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

//#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
//#include <cutil_gl_inline.h> // includes cuda_gl_interop.h
//#include <rendercheck_gl.h>

// CUDA Includes
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>


#include "cuda_utils.h"


typedef unsigned int uint;
typedef unsigned char uchar;

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f
#define num_sets    3

typedef struct data_set DataSet;
typedef struct data_set *DataSetPtr;

DataSetPtr current;

struct data_set {
    //values for loading data file
    char *volumeFilename;
    char *filePath;
    cudaExtent volumeSize;
    size_t size;
    void *h_volume;
    //VolumeType;
    //=======
    uint *d_output;
    size_t num_bytes;
    GLuint tex;
    GLuint pbo;
    struct cudaGraphicsResource *cuda_pbo_resource;    
    //values that go to render_kernel
    uint width;
    uint height;
    dim3 gridSize;
    dim3 blockSize;
    float density;
    float brightness;
    float transferOffset;
    float transferScale;
    bool linearFiltering;
    //==========
    float3 viewRotation;
    float3 viewTranslation;
};

DataSet ds[num_sets];


unsigned int timer = 0;
float invViewMatrix[12];


#define MAX(a,b) ((a > b) ? a : b)

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void initCuda(void *h_volume, cudaExtent volumeSize);
extern "C" void freeCudaBuffers();
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                              float density, float brightness, float transferOffset, float transferScale);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

void initPixelBuffer(DataSetPtr ds_pix);

// render image using CUDA
void render(DataSetPtr ds_rend)
{
    //size_t num_bytes;

    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    //uint *d_output;
    // map PBO to get CUDA device pointer
    HANDLE_ERROR( cudaGraphicsMapResources(1, &(ds_rend->cuda_pbo_resource), 0) );
    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&(ds_rend->d_output), 
							       &(ds_rend->num_bytes), 
							       ds_rend->cuda_pbo_resource));
    //fprintf(stderr, "CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    HANDLE_ERROR(cudaMemset(ds_rend->d_output, 0, ds_rend->width*ds_rend->height*4));

    // call CUDA kernel, writing results to PBO
    render_kernel(ds_rend->gridSize, ds_rend->blockSize, ds_rend->d_output, ds_rend->width, 
		  ds_rend->height, ds_rend->density, ds_rend->brightness, ds_rend->transferOffset, 
		  ds_rend->transferScale);

    //cutilCheckMsg("kernel failed");

    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &(ds_rend->cuda_pbo_resource), 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
    // use OpenGL to build view matrix
    int i;
    for(i=0; i<num_sets; i++)
    {
	GLfloat modelView[16];
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
        glLoadIdentity();
        glRotatef(-ds[i].viewRotation.x, 1.0, 0.0, 0.0);
        glRotatef(-ds[i].viewRotation.y, 0.0, 1.0, 0.0);
        glTranslatef(-ds[i].viewTranslation.x, -ds[i].viewTranslation.y, -ds[i].viewTranslation.z);
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	glPopMatrix();

	invViewMatrix[ 0] = modelView[ 0]; 
        invViewMatrix[ 1] = modelView[ 4]; 
	invViewMatrix[ 2] = modelView[ 8]; 
        invViewMatrix[ 3] = modelView[12];
	invViewMatrix[ 4] = modelView[ 1]; 
        invViewMatrix[ 5] = modelView[ 5]; 
	invViewMatrix[ 6] = modelView[ 9]; 
        invViewMatrix[ 7] = modelView[13];
	invViewMatrix[ 8] = modelView[ 2]; 
        invViewMatrix[ 9] = modelView[ 6]; 
	invViewMatrix[10] = modelView[10]; 
        invViewMatrix[11] = modelView[14];

	render(&(ds[i]));
    }
    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // draw using texture

    glEnable(GL_TEXTURE_2D);
    // copy from pbo to texture
    for(i=0; i<num_sets; i++)
    {
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, ds[i].pbo);
	glBindTexture(GL_TEXTURE_2D, ds[i].tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ds[i].width, ds[i].height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	// unbind pbo
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// draw textured quad
	glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(0, 0);
	glTexCoord2f(1, 0); glVertex2f(1, 0);
        glTexCoord2f(1, 1); glVertex2f(1, 1);
	glTexCoord2f(0, 1); glVertex2f(0, 1);
        glEnd();

	fprintf(stderr, "HERE\n");

    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    
    glutSwapBuffers();
    glutReportErrors();
}

void idle()
{
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    switch(key) {
        case 27:
            // escape key
            exit(0);
            break;
        case 'f':
            current->linearFiltering = !ds->linearFiltering;
            setTextureFilterMode(ds->linearFiltering);
            break;
        case '+':
            current->density += 0.01f;
            break;
        case '-':
            current->density -= 0.01f;
            break;

        case ']':
            current->brightness += 0.1f;
            break;
        case '[':
            current->brightness -= 0.1f;
            break;

        case ';':
            current->transferOffset += 0.01f;
            break;
        case '\'':
            current->transferOffset -= 0.01f;
            break;

        case '.':
            current->transferScale += 0.01f;
            break;
        case ',':
            current->transferScale -= 0.01f;
            break;

        default:
            break;
    }
    glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState  |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4) {
        // right = zoom
        current->viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 2) {
        // middle = translate
        current->viewTranslation.x += dx / 100.0f;
        current->viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 1) {
        // left = rotate
        current->viewRotation.x += dy / 5.0f;
        current->viewRotation.y += dx / 5.0f;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void reshape(int w, int h)
{
    int i;
    for(i=0; i<num_sets; i++)
    {
	ds[i].width = w; ds[i].height = h;
	initPixelBuffer(&(ds[i]));

    // calculate new grid size
	ds[i].gridSize = dim3(iDivUp(ds[i].width, ds[i].blockSize.x), iDivUp(ds[i].height, ds[i].blockSize.y));
    }

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup()
{
    freeCudaBuffers();
    int i;
    for(i=0; i<num_sets; i++)
    {
	if (ds[i].pbo) {
	    cudaGraphicsUnregisterResource(ds[i].cuda_pbo_resource);
	    glDeleteBuffersARB(1, &(ds[i].pbo));
	    glDeleteTextures(1, &(ds[i].tex));
	}
    }
}

void initGL(int *argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(ds->width, ds->height);
    glutCreateWindow("CUDA volume rendering");

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(1);
    }
}

void initPixelBuffer(DataSetPtr ds_pix)
{
    if (ds_pix->pbo) {
        // unregister this buffer object from CUDA C
        HANDLE_ERROR( cudaGraphicsUnregisterResource(ds_pix->cuda_pbo_resource) );

        // delete old buffer
        glDeleteBuffersARB(1, &ds_pix->pbo);
        glDeleteTextures(1, &ds_pix->tex);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &ds_pix->pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, ds_pix->pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, 
		    ds_pix->width*ds_pix->height*sizeof(GLubyte)*4, 0, 
		    GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    HANDLE_ERROR( cudaGraphicsGLRegisterBuffer(&ds_pix->cuda_pbo_resource, 
						ds_pix->pbo, 
						cudaGraphicsMapFlagsWriteDiscard) );

    // create texture for display
    glGenTextures(1, &ds_pix->tex);
    glBindTexture(GL_TEXTURE_2D, ds_pix->tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ds_pix->width, 
					     ds_pix->height, 0, 
					     GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// Load raw data from disk
void *loadRawFile(char *filename, size_t size)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    fprintf(stderr, "Read '%s', %d bytes\n", filename, read);

    return data;
}


////////////////////////////////////////////////////////////////////////////////
// Program main  just a test
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    cudaDeviceProp prop;
    int dev;

    HANDLE_ERROR(cudaGetDevice(&dev));
    fprintf(stderr, "ID of  current CUDA device: %d\n", dev);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 3;

    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
    printf("ID of CUDA device closest to revision 1.3: %d\n", dev);
    //HANDLE_ERROR(cudaSetDevice(dev));

    HANDLE_ERROR( cudaGLSetGLDevice(dev) );
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    initGL( &argc, argv );

//========= Initialize dataset structure values ===============================
    ds[0].volumeFilename = "Bucky.raw";
    ds[0].filePath   = "./data/Bucky.raw";
    ds[0].volumeSize = make_cudaExtent(32, 32, 32);
    ds[0].size = ds[0].volumeSize.width * ds[0].volumeSize.height * ds[0].volumeSize.depth * sizeof(unsigned char);
    ds[0].width  = 512;
    ds[0].height = 512;
    ds[0].blockSize	  = dim3(16, 16);
    ds[0].density	  = 0.05f;
    ds[0].brightness      = 1.0f;
    ds[0].transferOffset  = 0.0f;
    ds[0].transferScale   = 1.0f;
    ds[0].linearFiltering = true;
    ds[0].viewTranslation = make_float3(0.0, 0.0, -4.0f);
//========= End initializing dataset structure values =========================

//========= Initialize dataset structure values ===============================
    ds[1].volumeFilename = "Bucky.raw";
    ds[1].filePath   = "./data/Bucky.raw";
    ds[1].volumeSize = make_cudaExtent(32, 32, 32);
    ds[1].size   = ds[1].volumeSize.width * ds[1].volumeSize.height 
				      * ds[1].volumeSize.depth 
				      * sizeof(unsigned char);
    ds[1].width  = 512;
    ds[1].height = 512;
    ds[1].blockSize	  = dim3(16, 16);
    ds[1].density	  = 0.05f;
    ds[1].brightness      = 1.0f;
    ds[1].transferOffset  = 0.0f;
    ds[1].transferScale   = 1.0f;
    ds[1].linearFiltering = true;
    ds[1].viewTranslation = make_float3(0.0, 0.0, -4.0f);
//========= End initializing dataset structure values =========================

//========= Initialize dataset structure values ===============================
    ds[2].volumeFilename = "Bucky.raw";
    ds[2].filePath   = "./data/Bucky.raw";
    ds[2].volumeSize = make_cudaExtent(32, 32, 32);
    ds[2].size   = ds[2].volumeSize.width * ds[2].volumeSize.height 
				      * ds[2].volumeSize.depth 
				      * sizeof(unsigned char);
    ds[2].width  = 512;
    ds[2].height = 512;
    ds[2].blockSize	   = dim3(16, 16);
    ds[2].density	   = 0.05f;
    ds[2].brightness      = 1.0f;
    ds[2].transferOffset  = 0.0f;
    ds[2].transferScale   = 1.0f;
    ds[2].linearFiltering = true;
    ds[2].viewTranslation = make_float3(0.0, 0.0, -4.0f);
//========= End initializing dataset structure values =========================

    current = &(ds[1]);

    // load volume data
    int index;
    for(index=0; index<num_sets; index++)
    {
	ds[index].h_volume = loadRawFile(ds[index].filePath, ds[index].size);
	initCuda(ds[index].h_volume, ds[index].volumeSize);
	free(ds[index].h_volume);
	initPixelBuffer(&ds[index]);
    }
    // calculate new grid size
    current->gridSize = dim3(iDivUp(current->width, current->blockSize.x), 
			     iDivUp(current->height, current->blockSize.y));

    // This is the normal rendering path for VolumeRender
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);


    atexit(cleanup);

    glutMainLoop();

    //cutilDeviceReset();
}

