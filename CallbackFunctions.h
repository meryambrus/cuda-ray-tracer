#pragma once

#include "Scene.cuh"

extern const int WINDOW_WIDTH;
extern const int WINDOW_HEIGHT;


GPUProgram gpuProgram; // vertex and fragment shaders

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
    unsigned int vao;	// vertex array object id and texture id
    unsigned int textureId;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight)
    {
        glGenVertexArrays(1, &vao);	// create 1 vertex array object
        glBindVertexArray(vao);		// make it active

        unsigned int vbo;		// vertex buffer objects
        glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

        // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // sampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    void LoadTexture(const std::vector<vec4>& image) {
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_FLOAT, &image[0]);
    }

    void Draw() {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
        const unsigned int textureUnit = 0;
        if (location >= 0) {
            glUniform1i(location, textureUnit);
            glActiveTexture(GL_TEXTURE0 + textureUnit);
            glBindTexture(GL_TEXTURE_2D, textureId);
        }
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

FullScreenTexturedQuad* fullScreenTexturedQuad;
Scene* scene;

void onDisplay();

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	
	long timeStart = glutGet(GLUT_ELAPSED_TIME);

	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

    scene = new Scene();
    scene->build();
	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(WINDOW_WIDTH, WINDOW_HEIGHT);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
    onDisplay();
}

// Window has become invalid: Redraw
void onDisplay() {
    std::vector<vec4> image(WINDOW_WIDTH * WINDOW_HEIGHT);

    long timeStart = glutGet(GLUT_ELAPSED_TIME);

    //render here
    scene->render(image);

    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

    fullScreenTexturedQuad->LoadTexture(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    switch (key) {
    case 'w':
        scene->camera->move(0.01f, MoveDirection::FORWARD);
        break;
    case 's':
        scene->camera->move(0.01f, MoveDirection::BACKWARD);
        break;
    case 'a':
        scene->camera->move(0.01f, MoveDirection::LEFT);
        break;
    case 'd':
        scene->camera->move(0.01f, MoveDirection::RIGHT);
        break;
    case 'e':
        scene->camera->rotate(-(10.0f / 180.f) * M_PI, scene->camera->up);
        break;
    case 'q':
        scene->camera->rotate((10.0f / 180.f) * M_PI, scene->camera->up);
        break;
    }

}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {

}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {

}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    onDisplay();
}