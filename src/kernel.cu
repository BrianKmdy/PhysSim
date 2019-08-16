#include "kernel.cuh"

#include "Calc.h"

__global__
void calculatePositionGravity(Particle* particles, int nParticles, float timeElapsed, int step) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < nParticles; i += stride) {
		Vector* position = &particles[i].position[step % 2];
		Vector* lastPosition = &particles[i].position[(step + 1) % 2];

		v3 gravity = {0};

		for (int o = 0; o < nParticles; o++) {
			if (i != o) {
				Vector* otherPosition = &particles[i].position[(step + 1) % 2];

				float dist = sqrt(pow(otherPosition->y - lastPosition->y, 2.0f) + pow(otherPosition->x - lastPosition->x, 2.0f) + pow(otherPosition->z - lastPosition->z, 2.0f));
				if (dist > 0.0 && dist < MIN_DIST)
					dist = MIN_DIST;

				if (dist > 0.0) {
					float forceMagnitude = (G * (float)particles[i].mass * (float)particles[o].mass) / pow(dist, 2.0f);

					v3 v = {otherPosition->x - lastPosition->x, otherPosition->y - lastPosition->y, otherPosition->z - lastPosition->z};
					float vectorMagnitude = sqrt(pow(v.x, 2.0f) + pow(v.y, 2.0f) + pow(v.z, 2.0f));
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

		position->x = lastPosition->x + (particles[i].velocity.x * timeElapsed) + (0.5 * acceleration.x * pow(timeElapsed, 2.0f));
		position->y = lastPosition->y + (particles[i].velocity.y * timeElapsed) + (0.5 * acceleration.y * pow(timeElapsed, 2.0f));
		position->z = lastPosition->z + (particles[i].velocity.z * timeElapsed) + (0.5 * acceleration.z * pow(timeElapsed, 2.0f));

		particles[i].velocity.x = particles[i].velocity.x + (acceleration.x * timeElapsed);
		particles[i].velocity.y = particles[i].velocity.y + (acceleration.y * timeElapsed);
		particles[i].velocity.z = particles[i].velocity.z + (acceleration.z * timeElapsed);
	}
}

void doWork(Particle* particles, int nParticles, float timeElapsed, int step) {
	int blockSize = 256;
	int numBlocks = (nParticles + blockSize - 1) / blockSize;
	calculatePositionGravity <<<numBlocks, blockSize>>> (particles, nParticles, timeElapsed, step);
}

__global__
void calculatePositionHypothetical(Particle* particles, int nParticles, Particle* massiveParticles, int nMassiveParticles, float timeElapsed, v3 center, float radius, int step) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < nParticles; i += stride) {
		v3 force = { 0 };

		Vector* position = &particles[i].position[step % 2];
		Vector* lastPosition = &particles[i].position[(step + 1) % 2];

		// Add the force from the bounds
		float radiusFromCenter = sqrt(pow(lastPosition->x - center.x, 2.0f) + pow(lastPosition->y - center.y, 2.0f) + pow(lastPosition->z - center.z, 2.0f));
		float magnitude = pow(radiusFromCenter, 2.0f) * FORCE_CONST;
		v3 direction = { center.x - lastPosition->x, center.y - lastPosition->y, center.z - lastPosition->z };
		if (radiusFromCenter > 0.0f) {
			direction.x /= radiusFromCenter;
			direction.y /= radiusFromCenter;
			direction.z /= radiusFromCenter;
		}
		direction.x *= magnitude;
		direction.y *= magnitude;
		direction.z *= magnitude;

		force.x += direction.x;
		force.y += direction.y;
		force.z += direction.z;

		// Add the force from the particles in the medium
		for (int o = 0; o < nParticles; o++) {
			if (i != o) {
				Vector* otherPosition = &particles[o].position[(step + 1) % 2];

				float dist = sqrt(pow(otherPosition->y - lastPosition->y, 2.0f) + pow(otherPosition->x - lastPosition->x, 2.0f) + pow(otherPosition->z - lastPosition->z, 2.0f));
				v3 v = { lastPosition->x - otherPosition->x, lastPosition->y - otherPosition->y, lastPosition->z - otherPosition->z };
				float vectorMagnitude = sqrt(pow(v.x, 2.0f) + pow(v.y, 2.0f) + pow(v.z, 2.0f));
				if (dist > 0.01) {
					// XXX Review this, should be dist^2
					float forceMagnitude = (G * (float)particles[i].mass * (float)particles[o].mass) / pow(dist, 2.0f);
					if (vectorMagnitude > 0.0f) {
						v.x /= vectorMagnitude;
						v.y /= vectorMagnitude;
						v.z /= vectorMagnitude;
					}

					force.x += v.x * forceMagnitude;
					force.y += v.y * forceMagnitude;
					force.z += v.z * forceMagnitude;
				}
			}
		}

		// Now do it to the massive particles
		for (int o = 0; o < nMassiveParticles; o++) {
			Vector* otherPosition = &massiveParticles[o].position[(step + 1) % 2];

			float dist = sqrt(pow(otherPosition->y - lastPosition->y, 2.0f) + pow(otherPosition->x - lastPosition->x, 2.0f) + pow(otherPosition->z - lastPosition->z, 2.0f));
			if (dist > 0.0) {
				// XXX Review this, should be dist^2
				float forceMagnitude = (G * (float) particles[i].mass * (float) massiveParticles[o].mass) / pow(dist, 2.0f);

				v3 v = { lastPosition->x - otherPosition->x, lastPosition->y - otherPosition->y, lastPosition->z - otherPosition->z };
				float vectorMagnitude = sqrt(pow(v.x, 2.0f) + pow(v.y, 2.0f) + pow(v.z, 2.0f));
				if (vectorMagnitude > 0.0f) {
					v.x /= vectorMagnitude;
					v.y /= vectorMagnitude;
					v.z /= vectorMagnitude;
				}

				force.x += v.x * forceMagnitude;
				force.y += v.y * forceMagnitude;
				force.z += v.z * forceMagnitude;
			}
		}

		v3 acceleration = { force.x / particles[i].mass, force.y / particles[i].mass, force.z / particles[i].mass };

		float speedLimit = 20.0f;

		v3 moveVector;
		moveVector.x = (particles[i].velocity.x * timeElapsed) + (0.5 * acceleration.x * pow(timeElapsed, 2.0f));
		moveVector.y = (particles[i].velocity.y * timeElapsed) + (0.5 * acceleration.y * pow(timeElapsed, 2.0f));
		moveVector.z = (particles[i].velocity.z * timeElapsed) + (0.5 * acceleration.z * pow(timeElapsed, 2.0f));
		float moveMagnitude = sqrt(pow(moveVector.x, 2.0f) + pow(moveVector.y, 2.0f) + pow(moveVector.z, 2.0f));
		if (moveMagnitude <= speedLimit) {
			position->x = lastPosition->x + moveVector.x;
			position->y = lastPosition->y + moveVector.y;
			position->z = lastPosition->z + moveVector.z;
		} else {
			position->x = lastPosition->x + speedLimit / moveMagnitude * moveVector.x;
			position->y = lastPosition->y + speedLimit / moveMagnitude * moveVector.y;
			position->z = lastPosition->z + speedLimit / moveMagnitude * moveVector.z;
		}

		v3 newVelocity;
		newVelocity.x = particles[i].velocity.x + (acceleration.x * timeElapsed);
		newVelocity.y = particles[i].velocity.y + (acceleration.y * timeElapsed);
		newVelocity.z = particles[i].velocity.z + (acceleration.z * timeElapsed);

		float lastVelocityMagnitude = sqrt(pow(particles[i].velocity.x, 2.0f) + pow(particles[i].velocity.y, 2.0f) + pow(particles[i].velocity.z, 2.0f));
		float velocityMagnitude = sqrt(pow(newVelocity.x, 2.0f) + pow(newVelocity.y, 2.0f) + pow(newVelocity.z, 2.0f));

		if (velocityMagnitude <= speedLimit || velocityMagnitude < lastVelocityMagnitude) {
			particles[i].velocity.x = newVelocity.x;
			particles[i].velocity.y = newVelocity.y;
			particles[i].velocity.z = newVelocity.z;
		} else {
			particles[i].velocity.x = speedLimit / velocityMagnitude * newVelocity.x;
			particles[i].velocity.y = speedLimit / velocityMagnitude * newVelocity.y;
			particles[i].velocity.z = speedLimit / velocityMagnitude * newVelocity.z;
		}
	}

	for (int i = index; i < nMassiveParticles; i += stride) {
		Vector* position = &massiveParticles[i].position[step % 2];
		Vector* lastPosition = &massiveParticles[i].position[(step + 1) % 2];

		v3 force = { 0 };

		// Add the force from the bounds
		float radiusFromCenter = sqrt(pow(lastPosition->x - center.x, 2.0f) + pow(lastPosition->y - center.y, 2.0f) + pow(lastPosition->z - center.z, 2.0f));
		v3 direction = { center.x - lastPosition->x, center.y - lastPosition->y, center.z - lastPosition->z };
		float magnitude = pow(radiusFromCenter, 2.0f) * massiveParticles[i].mass * FORCE_CONST;
		if (radiusFromCenter > 0.0f)
		{
			direction.x /= radiusFromCenter;
			direction.y /= radiusFromCenter;
			direction.z /= radiusFromCenter;
		}

		force.x += direction.x * magnitude;
		force.y += direction.y * magnitude;
		force.z += direction.z * magnitude;

		// Add the force from the particles in the medium
		for (int o = 0; o < nParticles; o++) {
			Vector* otherPosition = &particles[o].position[(step + 1) % 2];

			float dist = sqrt(pow(otherPosition->y - lastPosition->y, 2.0f) + pow(otherPosition->x - lastPosition->x, 2.0f) + pow(otherPosition->z - lastPosition->z, 2.0f));
			if (dist > 0.0) {
				// XXX Review this, should be dist^2
				float forceMagnitude = (G * (float)massiveParticles[i].mass * (float)particles[o].mass) / pow(dist, 2.0f);

				v3 v = { lastPosition->x - otherPosition->x, lastPosition->y - otherPosition->y, lastPosition->z - otherPosition->z };
				float vectorMagnitude = sqrt(pow(v.x, 2.0f) + pow(v.y, 2.0f) + pow(v.z, 2.0f));
				if (vectorMagnitude > 0.0f) {
					v.x /= vectorMagnitude;
					v.y /= vectorMagnitude;
					v.z /= vectorMagnitude;
				}

				force.x += v.x * forceMagnitude;
				force.y += v.y * forceMagnitude;
				force.z += v.z * forceMagnitude;
			}
		}

		v3 acceleration = { force.x / massiveParticles[i].mass, force.y / massiveParticles[i].mass, force.z / massiveParticles[i].mass };

		position->x = lastPosition->x + (massiveParticles[i].velocity.x * timeElapsed) + (0.5 * acceleration.x * pow(timeElapsed, 2.0f));
		position->y = lastPosition->y + (massiveParticles[i].velocity.y * timeElapsed) + (0.5 * acceleration.y * pow(timeElapsed, 2.0f));
		position->z = lastPosition->z + (massiveParticles[i].velocity.z * timeElapsed) + (0.5 * acceleration.z * pow(timeElapsed, 2.0f));

		massiveParticles[i].velocity.x = massiveParticles[i].velocity.x + (acceleration.x * timeElapsed);
		massiveParticles[i].velocity.y = massiveParticles[i].velocity.y + (acceleration.y * timeElapsed);
		massiveParticles[i].velocity.z = massiveParticles[i].velocity.z + (acceleration.z * timeElapsed);
	}
}

void doWorkHypothetical(Particle* particles, int nParticles, Particle* massiveParticles, int nMassiveParticles, float timeElapsed, v3 center, float radius, int step) {
	int blockSize = 256;
	int numBlocks = (nParticles + blockSize - 1) / blockSize;
	calculatePositionHypothetical << <numBlocks, blockSize >> > (particles, nParticles, massiveParticles, nMassiveParticles, timeElapsed, center, radius, step);
}