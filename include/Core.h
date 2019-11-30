#pragma once

// I propose a new fundamental force as the possible source for gravity. I imagine that everywhere in space there is a fluid
// called the cosmic medium which is infinitely compressible and seeks to spread out as much as possible. What we recognize
// as gravity could simply be the absence of this force between bodies of mass. Note that this hypothesis is inspired by
// ideas such as dark energy and integrates the concept of an expanding universe
//
// Hypothesis of the universal medium
// 1. At the time of the big bang the cosmic medium was an infinitely compressed fluid
// 2. Between any infinitesimal division of the fluid there is a repulsive force causing it to spread apart
// 3. The fluid spread apart rapidly leading to areas of uniform density in the center of the universe and aread of
//    high density at the outer edges of the expanding universe
// 4. Over time as the density becomes uniform the fluid begins to condense into matter
// 5. Since matter is highly concentrated relative to the fluid its repulsive force is also much stronger
// 6. The highly condensed matter creates small bubbles or pockets around it with lower density fluid
// 7. Rather than gravity being an attractive force between matter it's actually just the lack of a repulsive force
//    as these pockets create low density areas of fluid between them
// 8. As the low density pockets of fluid form the matter is squeezed together by the fluid surrounding it
// 9. The fluid is also the medium through which electromagnic waves travel at the speed of light
// 10. Action at a distance is impossible (classical gravity), any force which acts over distance must travel through the medium
//
// Further testing needs to be done to prove the validity of this hypothesis
// ~Brian Kimball Moody
// 11/14/2019

#include <vector>
#include <chrono>
#include <filesystem>

#include "Simulate.cuh"
#include "Operations.cuh"
#include "Types.h"

void dumpParticles(std::string name, int nParticles, Particle* particles);
void dumpBoxes(std::string name, int nBoxes, Box* boxes);

class FrameBufferOut : public FrameBuffer<Particle>
{
public:
	FrameBufferOut(int queueSize, int nParticles, int stepSize);

	virtual void nextFrame(std::shared_ptr<Particle[]>* frame);
	virtual void run();
};

class Core
{
public:
	Core();
    Core(std::shared_ptr<Instance> instance, std::shared_ptr<Particle[]> particles, std::shared_ptr<Box[]> boxes);
	~Core();

	void verifyConfiguration();

	std::shared_ptr<Instance> getInstance();
	std::shared_ptr<Particle[]> getParticles();
	std::shared_ptr<Box[]> getBoxes();
	void setInstance(std::shared_ptr<Instance> instance);
	void setParticles(std::shared_ptr<Particle[]> particles);
	void setBoxes(std::shared_ptr<Box[]> boxes);

	void setFramesPerPosition(int framesPerPosition);
	void setFramesPerState(int framesPerState);

	void setKernel(std::string kernelName);

	void run();
	void kill();
private:

	void writePositionToDisk();
	void writeStateToDisk();
	std::chrono::milliseconds writeToDisk();

	volatile bool alive;

	std::shared_ptr<Instance> instance;
	std::shared_ptr<Particle[]> particles;
	std::shared_ptr<Box[]> boxes;

	int framesPerPosition;
	int framesPerState;

	std::shared_ptr<FrameBufferOut> frameBuffer;

	int kernel;
	std::string kernelName;

	int frame;
	std::chrono::milliseconds startTime;
	std::chrono::milliseconds frameTime;
};