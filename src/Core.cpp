#include "Core.h"

#include <chrono>
#include <thread>

#define MIN_DIST 5

Core::Core(double width, double height, bool interParticleGravity) {
    this->interParticleGravity = interParticleGravity;
    entities = std::vector<Particle>();

    this->ui = new GUI(width, height);
}

void Core::setRate(double rate) {
    this->rate = rate;
}

void Core::setG(double G) {
    this->G = G;
}

void Core::setGravity(Vector gravity) {
    this->gravity = gravity;
}

void Core::setOutput(int output) {
    this->output = output;
    ui->setOutput(output);
}

void Core::setFps(double fps) {
    this->fps = fps;
}

void Core::addEntity(Particle particle) {
    //particle.addConstantForce(Vector(gravity.getMagnitude() * particle.getMass(), gravity.getDirection()));
    entities.push_back(particle);
}

GUI *Core::getGUI() {
    return ui;
}

void Core::run() {
	std::chrono::nanoseconds startTime =
		std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());

    double timeElapsed = 0.0;

    while (true) {
        ui->tick(&entities);

        if (ui->shouldClose())
            break;

		std::chrono::nanoseconds currentTime =
			std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());

        int entsInFrame = 0;

        if (interParticleGravity) {
            if (entities.size() > 1) {
                std::vector<Particle *> significantEnts = std::vector<Particle *>();
                for (std::vector<Particle>::iterator it = entities.begin(); it < entities.end(); ++it) {
                    if (it->isSignificant())
                        significantEnts.push_back(&(*it));
                }

                for (std::vector<Particle>::iterator it = entities.begin(); it < entities.end(); ++it) {
                    Vector gravity = Vector();

                    for (std::vector<Particle *>::iterator it2 = significantEnts.begin(); it2 < significantEnts.end(); ++it2) {
                        if (&(*it) != &(**it2)) {
                            double dist = distance(*it, **it2);
                            if (dist > 0.0 && dist < MIN_DIST)
                                dist = MIN_DIST;

                            if (dist > 0.0) {
                                double magnitude = (G * (double) it->getMass() * (double) (*it2)->getMass()) / pow(dist, 2.0f);
                                Vector v = Vector((**it2).getX() - (*it).getX(), (**it2).getY() - (*it).getY(), (**it2).getZ() - (*it).getZ());
                                v = v.normalize();
                                v = v.product(magnitude);

                                gravity += v;
                            }
                        }
                    }

                    it->addVariableForce(gravity);
                }
            }
        }

        for (std::vector<Particle>::iterator it = entities.begin(); it < entities.end(); ++it) {
            if (output == GUI::OUTPUT_TO_SCREEN)
                it->think((currentTime - startTime).count() / 1000000000.0f * rate);
            else
                it->think((1.0 / fps) * rate);
        }
    }

    ui->tick(&entities);
    ui->terminate();

    delete ui;
}
