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

const int WINDOW_WIDTH = 1600;
const int WINDOW_HEIGHT = 900;
const float EPSILON = 1e-6;


int main(int argc, char* argv[]) {
    // Initialize GLUT, Glew and OpenGL 
    glutInit(&argc, argv);

    // OpenGL major and minor versions
    int majorVersion = 3, minorVersion = 3;

    glutInitContextVersion(majorVersion, minorVersion);
    
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitWindowPosition(200, 200);

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

    onInitialization();

    glutDisplayFunc(onDisplay);
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);

    glutMainLoop();
    return 0;
}