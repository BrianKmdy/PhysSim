#include "Core.h"

Core::Core(double width, double height, bool bounded, bool interParticleGravity) {
    bounds = Rectangle(0.0, 0.0, width, height);
    this->bounded = bounded;
    this->gravity = gravity;
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
    particle.addConstantForce(Vector(gravity.getMagnitude() * particle.getMass(), gravity.getDirection()));
    entities.push_back(particle);
}

void Core::run() {
    for (int i = 0; i < 1000000000; i++) {}

    timeval startTime;
    gettimeofday(&startTime, NULL);
    timeval lastTime = startTime;
    timeval currentTime;

    double timeElapsed = 0.0;

    while (true) {
        ui->tick(&entities);

        if (ui->shouldClose())
            break;

        gettimeofday(&currentTime, NULL);

        timeElapsed = (double) (currentTime.tv_sec - lastTime.tv_sec) + ((double) (currentTime.tv_usec - lastTime.tv_usec) / 1000000.0);
        lastTime = currentTime;

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
                            if (dist > 0.0 && dist < 1.0)
                                dist = 1.0;

                            if (dist > 0.0)
                                gravity += Vector((G * (double) it->getMass() * (double) (*it2)->getMass()) / pow(dist, 2.0f), direction(*it, **it2));
                        }
                    }

                    it->addVariableForce(gravity);
                }
            }
        }

        for (std::vector<Particle>::iterator it = entities.begin(); it < entities.end(); ++it) {
            if (output == GUI::OUTPUT_TO_SCREEN)
                it->think(timeElapsed * rate);
            else
                it->think((1.0 / fps) * rate);

            if (contains(bounds, *it))
                entsInFrame++;
        }

        if (entsInFrame == 0)
            break;
    }

    ui->tick(&entities);
    ui->terminate();

    delete ui;
}
