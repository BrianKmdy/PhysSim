#pragma once

#include <fstream>
#include <map>
#include <chrono>
#include <thread>

#include "yaml-cpp/yaml.h"

struct Kernel
{
	enum
	{
		unknown = 0,
		gravity,
		experimental
	};

	// static int fromString(std::string kernel);
	// static std::string toString(int kernel);

	static std::map<std::string, int> fromString;
	static std::map<int, std::string> toString;
};

// The current time in milliseconds
std::chrono::milliseconds getMilliseconds();

// Parse a YAML config to get the number of particles for the scenario
int getNParticles(YAML::Node* config);

// Function interface for reading/writing position or particle state data to file
void positionToFile(std::ofstream* file, float* x, float* y);
void positionFromFile(std::ifstream* file, float* x, float* y);
void particleToFile(std::ofstream* file, float* x, float* y, float* vx, float* vy, float* mass);
void particleFromFile(std::ifstream* file, float* x, float* y, float* vx, float* vy, float* mass);

// Generic class for buffering frames read or written from file
template <typename T>
class FrameBuffer
{
public:
	FrameBuffer(int queueSize, int nParticles, int stepSize):
		thread(nullptr),
		alive(true),
		bufferIndex(0),
		frameIndex(0),
		stepSize(stepSize)
	{
		framePool.reserve(queueSize);
	}

	virtual ~FrameBuffer()
	{
	}

	bool hasFramesBuffered() {
		return frameIndex < bufferIndex;
	}

	bool waitForFull() {
		return true;
	}

	int bufferSize() {
		return frames.size();
	}

	void start() {
		thread = std::make_shared<std::thread>(&FrameBuffer::run, this);
	}

	void stop() {
		alive = false;
	}

	void join() {
		if (thread) {
			thread->join();
			thread = nullptr;
		}
	}

	virtual void nextFrame(std::shared_ptr<T>* frame) = 0;
	virtual void run() = 0;

protected:
	std::shared_ptr<std::thread> thread;
	bool alive;

	int frameIndex;
	int bufferIndex;
	int stepSize;

	std::map<int, std::shared_ptr<T>> frames;
	std::vector<std::shared_ptr<T>> framePool;
};
