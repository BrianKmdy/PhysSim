#ifndef USERINTERFACE_H
#define USERINTERFACE_H

#include <vector>

#include "Particle.h"

class UserInterface
{
    protected:
        double wWidth;
        double wHeight;
        int width;
        int height;
        int selectedEntityIndex;

    public:
        UserInterface(double wWidth, double wHeight, int width = 0, int height = 0);

        virtual void tick(const std::vector<Particle> * entities) = 0;
        virtual void drawParticle(Particle particle) = 0;
        virtual bool shouldClose() = 0;
        virtual void terminate() = 0;
        virtual bool returnPressed() = 0;
};

#endif // USERINTERFACE_H
