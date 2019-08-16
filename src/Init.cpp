#include <ctime>
#include <iostream>
#include <cstdlib>

#define _USE_MATH_DEFINES
#include <math.h>

#include <windows.h>

#include "Particle.h"
#include "Core.h"
#include "Calc.h"

float get_rand(float max) {
	return ((static_cast<float>(rand()) / RAND_MAX) * 2.0 * max) - max;
}

void createGalaxy(Core *core, Vector position, Vector up, float radius, float mass, Vector velocity, float r, float g, float b, float variance, float starMaxRadius, float starMinRadius, int numStars, float minDistance) {
	Particle* particles;

	cudaError cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	cudaStatus = cudaMallocManaged(&particles, (numStars + 1) * sizeof(Particle));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	particles[0].setInitialPosition(position);
	particles[0].setVelocity(velocity);
	particles[0].setMass(mass);
	particles[0].setColor(r, g, b);

    Vector w = up.normalize();
    Vector u = up.orthogonal();
    Vector v = w.vProduct(u);

    for (int i = 1; i < numStars + 1; i++) {
        float theta = ((float) rand() / (float) RAND_MAX) * 2 * M_PI;
        float rad = ((float) rand() / (float) RAND_MAX) * (radius - minDistance) + minDistance;
        float starRadius = ((float) rand() / (float) RAND_MAX) * (starMaxRadius - starMinRadius) + starMinRadius;

        Vector uCoord = u.product(cos(theta)).product(rad);
        Vector vCoord = v.product(sin(theta)).product(rad);

        float starR = r + (((float) rand() / (float) RAND_MAX) * variance) * randNeg();
        float starG = g + (((float) rand() / (float) RAND_MAX) * variance) * randNeg();
        float starB = b + (((float) rand() / (float) RAND_MAX) * variance) * randNeg();

        Vector starPosition = position.sum(uCoord.sum(vCoord));
        float magnitude = sqrt((G * mass) / rad);
        Vector starVelocity = starPosition.difference(position).vProduct(up).normalize().product(magnitude).sum(velocity);

		particles[i].setInitialPosition(starPosition);
		particles[i].setVelocity(starVelocity);
		particles[i].setRadius(7);
		particles[i].setMass(rand() % 100 + 100);
		particles[i].setColor(starR, starG, starB);
    }

	core->addParticles(particles, numStars + 1);
}

void galaxy() {
    srand(time(NULL));

    Core core(2560, 1440, true);
	core.setTimeStep(0.00005);
	core.setStepsPerFrame(5);
	core.setFramesPerSecond(100);
	core.getGUI()->setFileName("galaxy_1000");
	core.setOutput(GUI::OUTPUT_TO_VIDEO);
    core.getGUI()->setCamera(Vector(0, -200, 1600), Vector(0, 0, 0), Vector(0, 1, 0));

    createGalaxy(&core, Vector(0, 50, 0), Vector(0, 1.5, 1), 2500, 1000000000, Vector(0, 0, 0), 0.7, 0.7, 1, 0.2, 2, 5, 1000, 30);

    core.run();
}

void cloud() {
	srand(time(NULL));

	Core core(2560, 1400, true);
	core.setTimeStep(0.025);
	core.setStepsPerFrame(1);
	core.setFramesPerSecond(100);
	core.getGUI()->setFileName("medium_220000_r2_2d_5");
	core.setOutput(GUI::OUTPUT_TO_VIDEO);
	core.getGUI()->setCamera(Vector(0, 0, 1800), Vector(0, 0, 0), Vector(0, 1, 0));

	core.center = { 0, 0, 0 };
	core.radius = 500;

	float radius = 1600.0;

	Particle* particles;
	const int numParticles = 220000;

//	Particle* massiveParticles;
//	const int numMassiveParticles = 0;

	cudaError cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	// Initialize the particles
	if (numParticles > 0)
	{
		cudaStatus = cudaMallocManaged(&particles, numParticles * sizeof(Particle));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
	}

//	// Initialize the massive particles
//	if (numMassiveParticles > 0)
//	{
//		cudaStatus = cudaMallocManaged(&massiveParticles, numMassiveParticles * sizeof(Particle));
//		if (cudaStatus != cudaSuccess) {
//			fprintf(stderr, "cudaMalloc failed!");
//			return;
//		}
//	}

	for (int i = 0; i < numParticles; i++) {
		while (true) {
			Vector v = Vector(get_rand(radius), get_rand(radius), 0);
			if (distance(v, Vector(0, 0, 0)) < radius)
			{
				particles[i].setInitialPosition(v);
				break;
			}
		}

		particles[i].setMass(2.0);
		particles[i].setColor(0, 0, 0);
		particles[i].setRadius(5);
	}

//	for (int i = 0; i < numMassiveParticles; i++) {
//		massiveParticles[i].setPosition(Vector(static_cast<float>(rand() % 200) - 100, static_cast<float>(rand() % 200) - 100, static_cast<float>(rand() % 200) - 100));
//
//		massiveParticles[i].setMass(5.0);
//		massiveParticles[i].setColor(1, 1, 1);
//		massiveParticles[i].setRadius(30);
//	}
//
	core.addParticles(particles, numParticles);
//	core.addMassiveParticles(massiveParticles, numMassiveParticles);

	core.run();
}

#ifdef USEWINDOWS
int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR lpCmdLine, int nCmdShow)
{
	galaxy();

    return 0;
}
#else
int main()
{
	cloud();

	return 0;
}
#endif
