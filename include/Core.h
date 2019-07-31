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

		Particle* massiveParticles;
		int nMassiveParticles;

        int output = GUI::OUTPUT_TO_SCREEN;

        double rate = 1.0;
        double G = 1.0;

		double timeStep;
		unsigned int stepsPerFrame;
		unsigned int framesPerSecond;

		unsigned long stepCount;

        Vector gravity = Vector();

public:
        Core(double width = 0.0, double height = 0.0, bool interParticleGravity = false);

        void setRate(double rate);
        void setG(double G);
        void setGravity(Vector gravity);

		void setTimeStep(double timeStep);
		void setStepsPerFrame(double stepsPerFrame);
		void setFramesPerSecond(double framesPerSecond);

        void setOutput(int output);

		void addEntities(Particle* particles, int nParticles);
		void addMassiveParticles(Particle* massiveParticles, int nMassiveParticles);
        void run();

        GUI *getGUI();

		v3 center;
		double radius;
};

#endif // CORE_H
