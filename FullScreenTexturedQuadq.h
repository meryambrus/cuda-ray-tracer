#pragma once

#include <vector>
#include "D:\C++\Libraries\glew-2.1.0\include\GL\glew.h"

#include "framework.h"

extern const int WINDOW_WIDTH;
extern const int WINDOW_HEIGHT;

class FullScreenTexturedQuadq {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
	GPUProgram* gpuProgram;
public:
	FullScreenTexturedQuadq(int windowWidth, int windowHeight, std::vector<vec4>& image, GPUProgram* _gpuProgram)
		: texture(WINDOW_WIDTH, WINDOW_HEIGHT, image)
	{
		gpuProgram = _gpuProgram;
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
	}
	
	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};