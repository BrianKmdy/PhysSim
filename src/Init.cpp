#include <ctime>
#include <iostream>
#include <cstdlib>

#include "Particle.h"
#include "Core.h"
#include "Calc.h"

using namespace std;
/*
void freefall() {
    Core core(100.0, 417.0, false);
    core.setGravity(Vector(9.80, 1.5 * M_PI));

    core.addEntity(Particle(50.0, 417.0, 1.0));

    core.run();
}

void explosion() {
    Core core(500.0, 400.0, false);
    core.setGravity(Vector(9.80, 1.5 * M_PI));

    for (double i = 0.0; i < 32.0; i++) {
        core.addEntity(Particle(250.0, 200.0, 1.0, Vector(40.0, M_PI * (i / 16.0))));
        core.addEntity(Particle(250.0, 200.0, 1.0, Vector(80.0, M_PI * (i / 16.0))));
        core.addEntity(Particle(250.0, 200.0, 1.0, Vector(120.0, M_PI * (i / 16.0))));
    }

    core.run();
}

void explosion2() {
    srand(time(NULL));

    Core core(500.0, 400.0, false);
    core.setGravity(Vector(9.80, 1.5 * M_PI));

    for (double i = 0.0; i < 400.0; i++)
        core.addEntity(Particle(250.0, 200.0, 1.0, Vector((rand() % 100) + (rand() % 50) + (rand() % 10), M_PI * ((float) (rand() % 5000) / 2500.0))));

    core.run();
}

void cloud() {
    srand(time(NULL));

    Core core(500.0, 400.0, false,true);

    for (double i = 0.0; i < 100.0; i++)
        core.addEntity(Particle(rand() % 500,rand() % 400, 1.0));

    core.run();
}

void cloud2() {
    srand(time(NULL));

    Core core(500.0, 400.0, false, true);

    for (double i = 0.0; i < 150.0; i++)
        core.addEntity(Particle(rand() % 500,rand() % 400, 10));

    core.addEntity(Particle(0.0, 200.0, 500000, Vector(100.0, 0)));

    core.run();
}

void cloud3() {
    srand(time(NULL));

    Core core(500.0, 400.0, false, true);

    for (double i = 0.0; i < 150.0; i++)
        core.addEntity(Particle(rand() % 500,rand() % 400, 10));

    core.addEntity(Particle(0.0, 200.0, 500000, Vector(25.0, 0)));

    core.run();
}

void star() {
    srand(time(NULL));

    Core core(500.0, 400.0, false, true);

    core.addEntity(Particle(250.0, 200.0, 500000));

    for (double i = 0.0; i < 150.0; i++)
        core.addEntity(Particle(400.0 + rand() % 100, 150.0 + rand() % 100, 1, Vector(rand() % 100, (0.75 * M_PI) + rand() % 2)));

    core.run();
}

void star2() {
    srand(time(NULL));

    Core core(500.0, 400.0, false, true);
    core.setOutput(GUI::OUTPUT_TO_VIDEO);
    core.setRate(0.5);

    core.addEntity(Particle(250.0, 200.0, 500000));

    const double num_particles = 2048.0;
    for (double i = 0.0; i < num_particles; i++) {
        double theta = i / ((2.0 * M_PI) / num_particles);
        Particle p = Particle(250.0 + (rand() % 150) * cos(theta), 200.0 + (rand() % 150) * sin(theta), rand() % 25, Vector(25 + rand() % 50, theta + (M_PI / 4.0)));
        p.setSignificant(false);
        core.addEntity(p);
    }

    core.run();
}

void star3() {
    srand(time(NULL));

    Core core(500.0, 400.0, false, true);

    core.addEntity(Particle(250.0, 200.0, 500000));

    const double num_particles = 64.0;
    for (double i = 0.0; i < num_particles; i++) {
        double theta = i / ((2.0 * M_PI) / num_particles);
        core.addEntity(Particle(250.0 + (50 + rand() % 100) * cos(theta), 200.0 + (50 + rand() % 100) * sin(theta), rand() % 25, Vector(25 + rand() % 50, theta + (M_PI * (3.0 /4.0)))));
    }

    core.run();
}

void uniform() {
    srand(time(NULL));

    Core core(500.0, 400.0, false, true);
    core.setOutput(GUI::OUTPUT_TO_VIDEO);
    core.setRate(0.01);

    core.addEntity(Particle(250.0, 200.0, 500000));

    const double num_particles = 2048.0;
    for (double i = 0.0; i < num_particles; i++) {
        double theta = (double) (rand() % (int) num_particles) * ((2 * M_PI) / num_particles);
        double radius = (double) (rand() % 200);

        double x = 250.0 + (radius * cos(theta));
        double y = 200.0 + (radius * sin(theta));

        double mass = (float) (rand() % 25);
        double force = (mass * 500000) / pow(radius, 2.0);
        double velocity = sqrt((radius * force) / mass);

        Particle p = Particle(x, y, mass, Vector(velocity, theta + (M_PI / 2.0)));
        p.setSignificant(false);
        core.addEntity(p);
    }

    core.run();
}

void binary() {
    srand(time(NULL));

    Core core(500.0, 400.0, false, true);

    core.addEntity(Particle(225.0, 200.0, 50000, Vector(25.0, M_PI / 2.0)));
    core.addEntity(Particle(275.0, 200.0, 50000, Vector(25.0, M_PI * (3.0 / 2.0))));

    core.run();
}

void binary2() {
    srand(time(NULL));

    Core core(500.0, 400.0, false, true);

    core.addEntity(Particle(225.0, 200.0, 50000, Vector(30.0, M_PI * (1.0 / 4.0))));
    core.addEntity(Particle(275.0, 200.0, 35000, Vector(30.0, M_PI * (5.0 / 4.0))));

    core.run();
}

void binary3() {
    srand(time(NULL));

    Core core(500.0, 400.0, false, true);

    core.addEntity(Particle(225.0, 200.0, 50000, Vector(25.0, M_PI / 2.0)));
    core.addEntity(Particle(275.0, 200.0, 50000, Vector(25.0, M_PI * (3.0 / 2.0))));

    for (double i = 0.0; i < 150.0; i++)
        core.addEntity(Particle(400.0 + rand() % 100, 150.0 + rand() % 100, 1, Vector(rand() % 100, (0.75 * M_PI) + rand() % 2)));

    core.run();
}

void solar() {
    srand(time(NULL));

    Core core(5.0e5, 5.0e5, false, true);
    core.setG(6.67e-11);
    core.setRate(500.0);

    core.addEntity(Particle(5.0e2, 5.0e2, 5.0e20));

    for (double i = 0.0; i < 150.0; i++)
        core.addEntity(Particle(400.0 + rand() % 100, 150.0 + rand() % 100, 1, Vector(rand() % 100, (0.75 * M_PI) + rand() % 2)));

    core.run();
}
*/
void galaxy() {
    srand(time(NULL));

    Core core(1920, 1080, true);
    core.setOutput(GUI::OUTPUT_TO_SCREEN);
    core.setRate(0.1);


    core.addEntity(Particle(0.0, 0.0, 0.0, 500000));

    const double num_particles = 1000.0;

    for (double o = 0.0; o < 2; o++) {
        for (double i = 0.0; i < num_particles; i++) {
            double theta = i * ((2.0 * M_PI) / num_particles) + (o * M_PI);
            double radius = i * (150.0 / num_particles) + 25.0;

            double x = (radius * cos(theta)) * randNeg();
            if (rand() % 2 == 1)
                x += rand() % 10;
            else
                x -= rand() % 10;

            double y = (radius * sin(theta)) * randNeg();
            if (rand() % 2 == 1)
                y +=rand() % 10;
            else
                y -= rand() % 10;

            double mass = (double) (rand() % 25);
            double force = (double) (mass * 500000) / pow(radius, 2.0);
            double velocity = sqrt((double) (radius * force) / (double) mass);

            Particle p = Particle(x, y, 0.0, mass, Vector(-velocity, theta + (M_PI / 2.0)));
            p.setSignificant(false);
            core.addEntity(p);
        }
    }

    core.run();
}

void galaxies() {
    srand(time(NULL));

    Core core(1920, 1080, true);
    core.setOutput(GUI::OUTPUT_TO_SCREEN);
    core.setRate(0.1);
    core.getGUI()->setCamera(Vector(0, 0, 200), Vector(0, 0, 0), Vector(0, 1, 0));

    core.addEntity(Particle(0.0, 0.0, 0.0, 500000, Vector(25.0, 0.0, 0.0)));

    const double num_particles = 1000.0;

    for (double o = 0.0; o < 2; o++) {
        for (double i = 0.0; i < num_particles; i++) {
            double theta = i * ((2.0 * M_PI) / num_particles) + (o * M_PI);
            double radius = i * (150.0 / num_particles) + 25.0;

            double x = 0.0 + (radius * cos(theta));
            if (rand() % 2 == 1)
                x += rand() % 10;
            else
                x -= rand() % 10;

            double y = 0.0 + (radius * sin(theta));
            if (rand() % 2 == 1)
                y +=rand() % 10;
            else
                y -= rand() % 10;

            double mass = (float) (rand() % 25);
            double force = (mass * 500000) / pow(radius, 2.0);
            double velocity = sqrt((radius * force) / mass);

            Vector v = Vector(-velocity, theta + (M_PI / 2.0), 0.0);
            Particle p = Particle(x, y, 0, mass, v + Vector(25.0, 0.0, 0.0));
            p.setSignificant(false);
            core.addEntity(p);
        }
    }

    core.addEntity(Particle(100.0, 0.0, 100.0, 500000, Vector(-25.0, 0.0, 0.0)));

    for (double o = 0.0; o < 2; o++) {
        for (double i = 0.0; i < num_particles; i++) {
            double theta = i * ((2.0 * M_PI) / num_particles) + (o * M_PI);
            double radius = i * (150.0 / num_particles) + 25.0;

            double x = 100.0 + (radius * cos(theta));
            if (rand() % 2 == 1)
                x += rand() % 10;
            else
                x -= rand() % 10;

            double y = 100.0 + (radius * sin(theta));
            if (rand() % 2 == 1)
                y +=rand() % 10;
            else
                y -= rand() % 10;

            double mass = (float) (rand() % 25);
            double force = (mass * 500000) / pow(radius, 2.0);
            double velocity = sqrt((radius * force) / mass);

            Vector v = Vector(-velocity, 0.0, theta + (M_PI / 2.0));
            Particle p = Particle(x, 0, y, mass, v + Vector(-25.0, 0.0, 0.0));
            p.setSignificant(false);
            core.addEntity(p);
        }
    }

    core.run();
}

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
        p.setColor(starR, starG, starB);
        p.setSignificant(false);
        core->addEntity(p);
    }
}

void galaxies1() {
    srand(time(NULL));

    Core core(1920, 1080, true);
    core.setOutput(GUI::OUTPUT_TO_VIDEO);
    core.setRate(0.2);
    core.getGUI()->setCamera(Vector(0, 0, 1200), Vector(0, 0, 0), Vector(0, 1, 0));

    createGalaxy(&core, Vector(0, 0, 0), Vector(0, 0, 1), 450, 1000000, Vector(0, 0, 0), 1, 1, 1, 0.3, 2, 5, 100000, 1, 30);
    createGalaxy(&core, Vector(500, 50, 0), Vector(0, 1, 1), 450, 1000000, Vector(-100, 0, 0), 0.5, 0.5, 1, 0.2, 2, 5, 100000, 1, 30);

    core.run();
}

int main() {
    galaxies1();

    return 0;
}
