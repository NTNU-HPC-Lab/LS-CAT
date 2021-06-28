#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Shader.h"

extern "C" unsigned int width;
extern "C" unsigned int height;

extern "C" int maxIteration;
extern "C" double middlea;
extern "C" double middleb;
extern "C" double rangea;
extern "C" double rangeb;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void calculate(GLFWwindow* window);
void calculationCPU(float* mandelbrot);

void uploadNewTexture(float* data, GLuint TEXTURE, GLuint PBO);
void setUp(GLuint& VAO, GLuint& VBO, GLuint& EBO, GLuint& PBO, GLuint& TEXTURE);
void updateTexture(GLuint TEXTURE, GLuint PBO);

extern "C" void setUpCuda(GLuint PBO);
extern "C" void launchCuda(GLuint PBO);