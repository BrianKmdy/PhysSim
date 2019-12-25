#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <deque>
#include <cstdio>

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

class Replayer
{
public:
	Replayer();

	void setShaderPath(std::filesystem::path shaderPath);
	void setSceneDirectory(std::filesystem::path sceneDirectory);
	void setFrameStep(int frameStep);
	void setSpeed(int speed);
	void setOutputToVideo(bool outputToVideo);
	void setCodec(std::string codec);
	void setFormat(std::string format);
	void setParticleRadius(float radius);

	bool init();
	void run();
	void shutdown();

private:
	void handleInput();
	void reset();
	void update();
	void refresh();

	bool alive;

	std::filesystem::path shaderPath;
	std::filesystem::path sceneDirectory;

	int frameStep;
	int speed;
	int interFrames;

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

	float particleRadius;

	unsigned int glFrameBuffer;

	bool outputToVideo;
	FILE* outputVideoPipe;
	std::string codec;
	std::string format;
};