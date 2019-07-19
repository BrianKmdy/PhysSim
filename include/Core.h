#ifndef CORE_H
#define CORE_H

#include <vector>
// #include <sys/time.h>
// #include <unistd.h>

#include "Particle.h"
#include "Rectangle.h"
#include "Vector.h"
#include "UserInterface.h"
#include "GUI.h"
#include "Calc.h"

#include "kernel.cuh"

#ifdef __unix__
    #include "TUI.h"
#endif

class Core
{
    private:
        bool interParticleGravity;
        
        GUI * ui;

		Particle* entities;
		int nParticles;

        int output = GUI::OUTPUT_TO_SCREEN;
        double fps = 30.0;

        double rate = 1.0;
        double G = 1.0;

        Vector gravity = Vector();

    public:
        Core(double width = 0.0, double height = 0.0, bool interParticleGravity = false);

        void setRate(double rate);
        void setG(double G);
        void setGravity(Vector gravity);

        void setOutput(int output);
        void setFps(double fps);

		void addEntities(Particle* particles, int nParticles);
        void run();

        GUI *getGUI();
};

#endif // CORE_H
