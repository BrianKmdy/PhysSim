#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Particle.h"

struct v3
{
	float x;
	float y;
	float z;
};

__global__
void calculatePositionGravity(Particle* particles, int nParticles, float timeElapsed, int step);
void doWork(Particle* particles, int nParticles, float timeElapsed, int step);

__global__
void calculatePositionHypothetical(Particle* particles, int nParticles, Particle* massiveParticles, int nMassiveParticles, float timeElapsed, v3 center, float radius, int step);
void doWorkHypothetical(Particle* particles, int nParticles, Particle* massiveParticles, int nMassiveParticles, float timeElapsed, v3 center, float radius, int step);