#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <deque>

#include "spdlog/spdlog.h"
#include "glm/ext.hpp"

// Base shader class code from https://learnopengl.com/Getting-started/Shaders
class Shader
{
public:
	unsigned int ID;

	Shader(const char* vertexPath, const char* fragmentPath);

	void use();

	void setBool(const std::string& name, bool value) const;
	void setInt(const std::string& name, int value) const;
	void setFloat(const std::string& name, float value) const;
	void setMatrix(const std::string& name, glm::mat4 value) const;
private:
	void checkCompileErrors(unsigned int shader, std::string type);
};

class FrameBufferIn : public FrameBuffer<glm::vec3[]>
{
public:
	FrameBufferIn(int queueSize, int nParticles, int stepSize, std::map<int, std::string> files);

	bool hasMoreFrames();

	virtual void nextFrame(std::shared_ptr<glm::vec3[]>* frame);
	virtual void run();

private:
	std::map<int, std::string> files;
};

class Processor
{
public:
	Processor();

	bool init();
	void run();
	void shutdown();

private:
	void handleInput();
	void refresh();

	bool alive;

	std::shared_ptr<glm::vec3[]> currentFrame;

	std::shared_ptr<Shader> shader;
	std::shared_ptr<FrameBufferIn> frameBuffer;

	unsigned int VBO = 0;
	unsigned int VAO = 0;
};