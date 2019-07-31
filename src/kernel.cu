#include "kernel.cuh"

#include "Calc.h"

#define MIN_DIST 0.1
#define G 1.0
#define FORCE_CONST 0.000001

__global__
void calculatePositionGravity(Particle* particles, int nParticles, double timeElapsed) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < nParticles; i += stride) {
		v3 gravity = {0};

		for (int o = 0; o < nParticles; o++) {
			if (i != o) {
				double dist = sqrt(pow(particles[o].y - particles[i].y, 2.0) + pow(particles[o].x - particles[i].x, 2.0) + pow(particles[o].z - particles[i].z, 2.0));
				if (dist > 0.0 && dist < MIN_DIST)
					dist = MIN_DIST;

				if (dist > 0.0) {
					double forceMagnitude = (G * (double)particles[i].mass * (double)particles[o].mass) / pow(dist, 2.0);

					v3 v = {particles[o].x - particles[i].x, particles[o].y - particles[i].y, particles[o].z - particles[i].z};
					double vectorMagnitude = sqrt(pow(v.x, 2.0) + pow(v.y, 2.0) + pow(v.z, 2.0));
					v.x /= vectorMagnitude;
					v.y /= vectorMagnitude;
					v.z /= vectorMagnitude;

					gravity.x += v.x * forceMagnitude;
					gravity.y += v.y * forceMagnitude;
					gravity.z += v.z * forceMagnitude;
				}
			}
		}

		v3 acceleration = { gravity.x / particles[i].mass, gravity.y / particles[i].mass, gravity.z / particles[i].mass };

		particles[i].x = particles[i].x + (particles[i].velocity.x * timeElapsed) + (0.5 * acceleration.x * pow(timeElapsed, 2.0));
		particles[i].y = particles[i].y + (particles[i].velocity.y * timeElapsed) + (0.5 * acceleration.y * pow(timeElapsed, 2.0));
		particles[i].z = particles[i].z + (particles[i].velocity.z * timeElapsed) + (0.5 * acceleration.z * pow(timeElapsed, 2.0));

		particles[i].velocity.x = particles[i].velocity.x + (acceleration.x * timeElapsed);
		particles[i].velocity.y = particles[i].velocity.y + (acceleration.y * timeElapsed);
		particles[i].velocity.z = particles[i].velocity.z + (acceleration.z * timeElapsed);
	}
}

void doWork(Particle* particles, int nParticles, double timeElapsed) {
	int blockSize = 256;
	int numBlocks = (nParticles + blockSize - 1) / blockSize;
	calculatePositionGravity <<<numBlocks, blockSize>>> (particles, nParticles, timeElapsed);
}

__global__
void calculatePositionHypothetical(Particle* particles, int nParticles, Particle* massiveParticles, int nMassiveParticles, double timeElapsed, v3 center, double radius) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < nParticles; i += stride) {
		v3 force = { 0 };

		// Add the force from the bounds
		
		double radiusFromCenter = sqrt(pow(particles[i].x - center.x, 2.0) + pow(particles[i].y - center.y, 2.0) + pow(particles[i].z - center.z, 2.0));
		double magnitude = pow(radiusFromCenter, 2.0) * FORCE_CONST;
		v3 direction = { center.x - particles[i].x, center.y - particles[i].y, center.z - particles[i].z };
		direction.x /= radiusFromCenter;
		direction.y /= radiusFromCenter;
		direction.z /= radiusFromCenter;
		direction.x *= magnitude;
		direction.y *= magnitude;
		direction.z *= magnitude;

		force.x += direction.x;
		force.y += direction.y;
		force.z += direction.z;

		// Add the force from the particles in the medium
		for (int o = 0; o < nParticles; o++) {
			if (i != o) {
				double dist = sqrt(pow(particles[o].y - particles[i].y, 2.0) + pow(particles[o].x - particles[i].x, 2.0) + pow(particles[o].z - particles[i].z, 2.0));
				if (dist > 0.0 && dist < MIN_DIST)
					dist = MIN_DIST;

				if (dist > 0.0) {
					// XXX Review this, should be dist^2
					double forceMagnitude = (G * (double)particles[i].mass * (double)particles[o].mass) / pow(dist, 2.0);

					v3 v = { particles[i].x - particles[o].x, particles[i].y - particles[o].y, particles[i].z - particles[o].z };
					double vectorMagnitude = sqrt(pow(v.x, 2.0) + pow(v.y, 2.0) + pow(v.z, 2.0));
					v.x /= vectorMagnitude;
					v.y /= vectorMagnitude;
					v.z /= vectorMagnitude;

					force.x += v.x * forceMagnitude;
					force.y += v.y * forceMagnitude;
					force.z += v.z * forceMagnitude;
				}
			}

			// Now do it to the massive particles
			for (int o = 0; o < nMassiveParticles; o++) {
				double dist = sqrt(pow(massiveParticles[o].y - particles[i].y, 2.0) + pow(massiveParticles[o].x - particles[i].x, 2.0) + pow(massiveParticles[o].z - particles[i].z, 2.0));
				if (dist > 0.0 && dist < MIN_DIST)
					dist = MIN_DIST;

				if (dist > 0.0) {
					// XXX Review this, should be dist^2
					double forceMagnitude = (G * (double)particles[i].mass * (double)massiveParticles[o].mass) / pow(dist, 2.0);

					v3 v = { particles[i].x - massiveParticles[o].x, particles[i].y - massiveParticles[o].y, particles[i].z - massiveParticles[o].z };
					double vectorMagnitude = sqrt(pow(v.x, 2.0) + pow(v.y, 2.0) + pow(v.z, 2.0));
					v.x /= vectorMagnitude;
					v.y /= vectorMagnitude;
					v.z /= vectorMagnitude;

					force.x += v.x * forceMagnitude;
					force.y += v.y * forceMagnitude;
					force.z += v.z * forceMagnitude;
				}
			}
		}

		v3 acceleration = { force.x / particles[i].mass, force.y / particles[i].mass, force.z / particles[i].mass };

		particles[i].x = particles[i].x + (particles[i].velocity.x * timeElapsed) + (0.5 * acceleration.x * pow(timeElapsed, 2.0));
		particles[i].y = particles[i].y + (particles[i].velocity.y * timeElapsed) + (0.5 * acceleration.y * pow(timeElapsed, 2.0));
		particles[i].z = particles[i].z + (particles[i].velocity.z * timeElapsed) + (0.5 * acceleration.z * pow(timeElapsed, 2.0));

		particles[i].velocity.x = particles[i].velocity.x + (acceleration.x * timeElapsed);
		particles[i].velocity.y = particles[i].velocity.y + (acceleration.y * timeElapsed);
		particles[i].velocity.z = particles[i].velocity.z + (acceleration.z * timeElapsed);
	}

	for (int i = index; i < nMassiveParticles; i += stride) {
		v3 force = { 0 };

		// Add the force from the bounds

		double radiusFromCenter = sqrt(pow(massiveParticles[i].x - center.x, 2.0) + pow(massiveParticles[i].y - center.y, 2.0) + pow(massiveParticles[i].z - center.z, 2.0));
		v3 direction = { center.x - massiveParticles[i].x, center.y - massiveParticles[i].y, center.z - massiveParticles[i].z };
		direction.x /= radiusFromCenter;
		direction.y /= radiusFromCenter;
		direction.z /= radiusFromCenter;
		direction.x *= FORCE_CONST;
		direction.y *= FORCE_CONST;
		direction.z *= FORCE_CONST;

		force.x += direction.x;
		force.y += direction.y;
		force.z += direction.z;

		// Add the force from the particles in the medium
		for (int o = 0; o < nParticles; o++) {
			if (i != o) {
				double dist = sqrt(pow(particles[o].y - massiveParticles[i].y, 2.0) + pow(particles[o].x - massiveParticles[i].x, 2.0) + pow(particles[o].z - massiveParticles[i].z, 2.0));
				if (dist > 0.0 && dist < MIN_DIST)
					dist = MIN_DIST;

				if (dist > 0.0) {
					// XXX Review this, should be dist^2
					double forceMagnitude = (G * (double)massiveParticles[i].mass * (double)particles[o].mass) / pow(dist, 2.0);

					v3 v = { massiveParticles[i].x - particles[o].x, massiveParticles[i].y - particles[o].y, massiveParticles[i].z - particles[o].z };
					double vectorMagnitude = sqrt(pow(v.x, 2.0) + pow(v.y, 2.0) + pow(v.z, 2.0));
					v.x /= vectorMagnitude;
					v.y /= vectorMagnitude;
					v.z /= vectorMagnitude;

					force.x += v.x * forceMagnitude;
					force.y += v.y * forceMagnitude;
					force.z += v.z * forceMagnitude;
				}
			}
		}

		v3 acceleration = { force.x / massiveParticles[i].mass, force.y / massiveParticles[i].mass, force.z / massiveParticles[i].mass };

		massiveParticles[i].x = massiveParticles[i].x + (massiveParticles[i].velocity.x * timeElapsed) + (0.5 * acceleration.x * pow(timeElapsed, 2.0));
		massiveParticles[i].y = massiveParticles[i].y + (massiveParticles[i].velocity.y * timeElapsed) + (0.5 * acceleration.y * pow(timeElapsed, 2.0));
		massiveParticles[i].z = massiveParticles[i].z + (massiveParticles[i].velocity.z * timeElapsed) + (0.5 * acceleration.z * pow(timeElapsed, 2.0));

		massiveParticles[i].velocity.x = massiveParticles[i].velocity.x + (acceleration.x * timeElapsed);
		massiveParticles[i].velocity.y = massiveParticles[i].velocity.y + (acceleration.y * timeElapsed);
		massiveParticles[i].velocity.z = massiveParticles[i].velocity.z + (acceleration.z * timeElapsed);
	}
}

void doWorkHypothetical(Particle* particles, int nParticles, Particle* massiveParticles, int nMassiveParticles, double timeElapsed, v3 center, double radius) {
	int blockSize = 256;
	int numBlocks = (nParticles + blockSize - 1) / blockSize;
	calculatePositionHypothetical << <numBlocks, blockSize >> > (particles, nParticles, massiveParticles, nMassiveParticles, timeElapsed, center, radius);
}