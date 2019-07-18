#include <ctime>
#include <iostream>
#include <cstdlib>

#include <windows.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "Particle.h"
#include "Core.h"
#include "Calc.h"

void createGalaxy(Core *core, Vector position, Vector up, double radius, double mass, Vector velocity, double r, double g, double b, double variance, double starMaxRadius, double starMinRadius, int numStars, double G, double minDistance) {
    Particle c = Particle(position.getX(), position.getY(), position.getZ(), mass, velocity, 60);
    c.setColor(r, g, b);
    core->addEntity(c);

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
        Particle p = Particle(starPosition, 1, starVelocity, starRadius);
	    if (i % 30 == 0)
	    {
	    	Particle p2 = Particle(starPosition, 10000000, starVelocity, starRadius);
			p2.setColor(255, 0, 0);
	    	core->addEntity(p2);
	    }
        p.setColor(starR, starG, starB);
        // p.setSignificant(false);
        core->addEntity(p);
    }
}

void galaxies() {
    srand(time(NULL));

    Core core(2560, 1440, true);
    core.setOutput(GUI::OUTPUT_TO_SCREEN);
    core.setRate(0.05);
    core.setFps(60);
    core.getGUI()->setCamera(Vector(0, 0, 1200), Vector(0, 0, 0), Vector(0, 1, 0));

    // createGalaxy(&core, Vector(-750, 0, 0), Vector(0, 0, 1), 450, 1000000000, Vector(0, 0, 0), 1, 1, 1, 0.3, 2, 5, 100, 1, 30);
    createGalaxy(&core, Vector(0, 50, 0), Vector(0, 1.5, 1), 450, 1000000000, Vector(0, 0, 0), 0.7, 0.7, 1, 0.2, 2, 5, 250, 1, 30);

    core.run();
}

void binary() {
    srand(time(NULL));

    Core core(500.0, 400.0, true);
    core.setOutput(GUI::OUTPUT_TO_SCREEN);
    core.setRate(0.25);
    core.setFps(60);
    core.getGUI()->setCamera(Vector(0, 0, 200), Vector(0, 0, 0), Vector(0, 1, 0));

    core.addEntity(Particle(50, 0, 0, 50000, Vector(0, 10, 10)));
    core.addEntity(Particle(-50, 0, 0, 50000));

    core.run();
}

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR lpCmdLine, int nCmdShow)
{
	galaxies();

    return 0;
}
