#include <ctime>
#include <iostream>
#include <cstdlib>

#define _USE_MATH_DEFINES
#include <math.h>

#include "Particle.h"
#include "Core.h"
#include "Calc.h"

void createGalaxy(Core *core, Vector position, Vector up, double radius, double mass, Vector velocity, double r, double g, double b, double variance, double starMaxRadius, double starMinRadius, int numStars, double G, double minDistance) {
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

	particles[0].setPosition(position);
	particles[0].setVelocity(velocity);
	particles[0].setMass(mass);
	particles[0].setColor(r, g, b);

    Vector w = up.normalize();
    Vector u = up.orthogonal();
    Vector v = w.vProduct(u);

    for (int i = 0; i < numStars; i++) {
        double theta = ((double) rand() / (double) RAND_MAX) * 2 * M_PI;
        double rad = ((double) rand() / (double) RAND_MAX) * (radius - minDistance) + minDistance;
        double starRadius = ((double) rand() / (double) RAND_MAX) * (starMaxRadius - starMinRadius) + starMinRadius;

        Vector uCoord = u.product(cos(theta)).product(rad);
        Vector vCoord = v.product(sin(theta)).product(rad);

        double starR = r + (((double) rand() / (double) RAND_MAX) * variance) * randNeg();
        double starG = g + (((double) rand() / (double) RAND_MAX) * variance) * randNeg();
        double starB = b + (((double) rand() / (double) RAND_MAX) * variance) * randNeg();

        Vector starPosition = position.sum(uCoord.sum(vCoord));
        double magnitude = sqrt((G * mass) / rad);
        Vector starVelocity = starPosition.difference(position).vProduct(up).normalize().product(magnitude).sum(velocity);

		particles[i].setPosition(starPosition);
		particles[i].setVelocity(starVelocity);
		
	    // if (i % 30 == 0)
	    // {
		// 	particles[i].setMass(10000000);
		// 	particles[i].setColor(255, 0, 0);
	    // }
		// else
		{
			particles[i].setMass(1);
			particles[i].setColor(starR, starG, starB);
		}
    }

	core->addEntities(particles, numStars + 1);
}

void galaxy() {
    srand(time(NULL));

    Core core(2560, 1440, true);
    core.setOutput(GUI::OUTPUT_TO_SCREEN);
    core.setRate(0.05);
    core.setFps(60);
    core.getGUI()->setCamera(Vector(0, 0, 1200), Vector(0, 0, 0), Vector(0, 1, 0));

    createGalaxy(&core, Vector(0, 50, 0), Vector(0, 1.5, 1), 450, 1000000000, Vector(0, 0, 0), 0.7, 0.7, 1, 0.2, 2, 5, 5000, 1, 30);

    core.run();
}

int main()
{
	galaxy();

    return 0;
}
