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


uint width = 512;
uint height = 512;
dim3 blockSize(16, 16);
dim3 gridSize;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

unsigned int timer = 0;


#define MAX(a,b) ((a > b) ? a : b)

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void initCuda(void *h_volume, cudaExtent volumeSize);
extern "C" void freeCudaBuffers();
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                              float density, float brightness, float transferOffset, float transferScale);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

void initPixelBuffer();

// render image using CUDA
void render()
{
    size_t num_bytes;

    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    HANDLE_ERROR( cudaGraphicsMapResources(1, &cuda_pbo_resource, 0) );
    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, cuda_pbo_resource));
    //fprintf(stderr, "CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    HANDLE_ERROR(cudaMemset(d_output, 0, width*height*4));

    // call CUDA kernel, writing results to PBO
    render_kernel(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale);

    //cutilCheckMsg("kernel failed");

    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
        glLoadIdentity();
        glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
        glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
        glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[ 2] = modelView[ 8]; invViewMatrix[ 3] = modelView[12];
    invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[ 6] = modelView[ 9]; invViewMatrix[ 7] = modelView[13];
    invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#if 0
    // draw using glDrawPixels (slower)
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
    // draw using texture

    // copy from pbo to texture
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(1, 0);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
#endif

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
            linearFiltering = !linearFiltering;
            setTextureFilterMode(linearFiltering);
            break;
        case '+':
            density += 0.01f;
            break;
        case '-':
            density -= 0.01f;
            break;

        case ']':
            brightness += 0.1f;
            break;
        case '[':
            brightness -= 0.1f;
            break;

        case ';':
            transferOffset += 0.01f;
            break;
        case '\'':
            transferOffset -= 0.01f;
            break;

        case '.':
            transferScale += 0.01f;
            break;
        case ',':
            transferScale -= 0.01f;
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
        viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 2) {
        // middle = translate
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 1) {
        // left = rotate
        viewRotation.x += dy / 5.0f;
        viewRotation.y += dx / 5.0f;
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
    width = w; height = h;
    initPixelBuffer();

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

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

    if (pbo) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &tex);
    }
}

void initGL(int *argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA volume rendering");

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(1);
    }
}

void initPixelBuffer()
{
    if (pbo) {
        // unregister this buffer object from CUDA C
        HANDLE_ERROR( cudaGraphicsUnregisterResource(cuda_pbo_resource) );

        // delete old buffer
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    HANDLE_ERROR( cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard) );

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
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

    // load volume data
    char * path = "./data/Bucky.raw";
    cudaExtent volumeSize = make_cudaExtent(32, 32, 32);

    size_t size = volumeSize.width * volumeSize.height * volumeSize.depth * sizeof(unsigned char);
    void *h_volume = loadRawFile(path, size);

    initCuda(h_volume, volumeSize);
    free(h_volume);

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    // This is the normal rendering path for VolumeRender
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    initPixelBuffer();

    atexit(cleanup);

    glutMainLoop();

    //cutilDeviceReset();
}

