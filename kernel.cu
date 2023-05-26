#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "helper_cuda.h"
#include "helper_math_functions.h"
#include "CallbackFunctions.h"
#include "D:\C++\Libraries\glew-2.1.0\include\GL\glew.h"
#include "D:\C++\Libraries\freeglut\include\GL\freeglut.h"
#include <vector>


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

const int WINDOW_WIDTH = 1600;
const int WINDOW_HEIGHT = 900;
const float EPSILON = 1e-6;


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void changeValueTest(int* ar, const int* a) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    ar[i * WINDOW_HEIGHT + j] = a[i * WINDOW_HEIGHT + j]+1;
}

int main(int argc, char* argv[]) {
    // Initialize GLUT, Glew and OpenGL 
    glutInit(&argc, argv);

    // OpenGL major and minor versions
    int majorVersion = 3, minorVersion = 3;

    glutInitContextVersion(majorVersion, minorVersion);
    
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);				// Application window is initially of resolution 600x600
    glutInitWindowPosition(200, 200);							// Relative location of the application window

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

    glutCreateWindow(argv[0]);

    glewExperimental = true;	// magic
    glewInit();

    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    cudaSetDevice(0);
    size_t size;
    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
    printf("cudaLimitMallocHeapSize = %d\n", size);

    // Initialize this program and create shaders
    onInitialization();

    glutDisplayFunc(onDisplay);                // Register event handlers
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);

    glutMainLoop();
    return 5;
}

int mainn()
{

    int deviceID;
    cudaDeviceProp props;

    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&props, deviceID);

    int CudaCores = _ConvertSMVer2Cores(props.major, props.minor) * props.multiProcessorCount;

    printf("Number of CUDA cores: %d\n", CudaCores);


    int blockSize = getGreatestCommonFactor(WINDOW_WIDTH, WINDOW_HEIGHT);
    while (blockSize * blockSize > 1024) {
        blockSize /= 2;
    }

    dim3 threadsPerBlock(blockSize, blockSize);
    printf("threadsPerBlock: %d, %d\n", threadsPerBlock.x, threadsPerBlock.y);

    dim3 numBlocks(WINDOW_WIDTH / threadsPerBlock.x, WINDOW_HEIGHT / threadsPerBlock.y);
    printf("numBlocks: %d, %d\n", numBlocks.x, numBlocks.y);

    int* a = new int[WINDOW_WIDTH * WINDOW_HEIGHT];

    for (int i = 0; i < WINDOW_WIDTH; i++) {
        for (int j = 0; j < WINDOW_HEIGHT; j++) {
            a[i * WINDOW_HEIGHT + j] = 0;
        }
    }
    
    int* dev_a = 0;
    int* dev_c = 0;

    cudaError_t cudaStatus = cudaMalloc((void**)&dev_a, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc2 failed!");
        return 1;
    }

    cudaStatus = cudaMemcpy(dev_a, a, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 1;
    }

    changeValueTest<<<numBlocks, threadsPerBlock>>>(dev_c, dev_a);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "changeValueTest launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    int* result = new int[WINDOW_WIDTH * WINDOW_HEIGHT];

    cudaMemcpy(result, dev_c, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 1;
    }

    int counter = 0;
    int anti_counter = 0;

    for (int i = 0; i < WINDOW_WIDTH; i++) {
        for (int j = 0; j < WINDOW_HEIGHT; j++) {
            if (result[i * WINDOW_HEIGHT + j] != 1) {
                //printf("Error!\n");
                //printf("%d\n", result[i * 1080 + j]);
                counter++;
            }
            else {
                anti_counter++;
            }
        }
    }

    printf("Jó: %d\nRossz: %d\n", anti_counter, counter);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
