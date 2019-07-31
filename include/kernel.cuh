#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Particle.h"

struct v3
{
	double x;
	double y;
	double z;
};

__global__
void calculatePositionGravity(Particle* particles, int nParticles, double timeElapsed);
void doWork(Particle* particles, int nParticles, double timeElapsed);

__global__
void calculatePositionHypothetical(Particle* particles, int nParticles, Particle* massiveParticles, int nMassiveParticles, double timeElapsed, v3 center, double radius);
void doWorkHypothetical(Particle* particles, int nParticles, Particle* massiveParticles, int nMassiveParticles, double timeElapsed, v3 center, double radius);