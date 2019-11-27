#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "spdlog/spdlog.h"
#include "glm/ext.hpp"

class Shader
{
public:
	unsigned int ID;
	// constructor generates the shader on the fly
	// ------------------------------------------------------------------------
	Shader(const char* vertexPath, const char* fragmentPath);
	// activate the shader
	// ------------------------------------------------------------------------
	void use();
	// utility uniform functions
	// ------------------------------------------------------------------------
	void setBool(const std::string& name, bool value) const;
	// ------------------------------------------------------------------------
	void setInt(const std::string& name, int value) const;
	// ------------------------------------------------------------------------
	void setFloat(const std::string& name, float value) const;
private:
	// utility function for checking shader compilation/linking errors.
	// ------------------------------------------------------------------------
	void checkCompileErrors(unsigned int shader, std::string type);
};

class Processor
{
public:
	bool init();
	void run();
	void shutdown();

private:
	void handleInput();
	void loadPosition();
	void refresh();

	bool alive;
	Shader* ourShader;

	unsigned int VBO = 0;
	unsigned int VAO = 0;

	unsigned int modelMatrix = 0;
	unsigned int viewMatrix = 0;
	unsigned int projectionMatrix = 0;
};