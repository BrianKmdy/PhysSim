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

	Shader(std::filesystem::path vertexPath, std::filesystem::path fragmentPath);

	void use();

	void setBool(const std::string& name, bool value) const;
	void setInt(const std::string& name, int value) const;
	void setFloat(const std::string& name, float value) const;
	void setMatrix(const std::string& name, glm::mat4 value) const;
private:
	void checkCompileErrors(unsigned int shader, std::string type);
};

class FrameBufferIn : public FrameBuffer<glm::vec3>
{
public:
	FrameBufferIn(int queueSize, int nParticles, int stepSize, std::map<int, std::string> files);

	int stepSize;

	bool hasMoreFrames();

	virtual void nextFrame(std::shared_ptr<glm::vec3[]>* frame);
	virtual void run();

private:
	std::map<int, std::string> files;
};

class Processor
{
public:
	Processor(int frameStep, float speed);

	bool init(std::filesystem::path shaderPath, std::filesystem::path sceneDirectory);
	void run();
	void shutdown();

private:
	void handleInput();
	void reset();
	void update();
	void refresh();

	bool alive;

	int frameStep;
	int interFrames;
	int bufferSize;

	int frame;

	std::shared_ptr<glm::vec3[]> currentFrame;
	std::shared_ptr<glm::vec3[]> nextFrame;

	std::shared_ptr<FrameBufferIn> frameBuffer;

	std::shared_ptr<Shader> shader;
	std::shared_ptr<Shader> controls;

	unsigned int VBOCurrent;
	unsigned int VBONext;
	unsigned int VAO;

	unsigned int VBOControls;
	unsigned int VAOControls;
};