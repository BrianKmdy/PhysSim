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
        
        GUI* ui;

		Particle* particles;
		int nParticles;

		Particle* massiveParticles;
		int nMassiveParticles;

        int output = GUI::OUTPUT_TO_SCREEN;

		float timeStep;
		unsigned int stepsPerFrame;
		unsigned int framesPerSecond;

		unsigned long stepCount;

        Vector gravity = Vector();

public:
        Core(float width = 0.0, float height = 0.0, bool interParticleGravity = false);

        void setRate(float rate);
        void setGravity(Vector gravity);

		void setTimeStep(float timeStep);
		void setStepsPerFrame(float stepsPerFrame);
		void setFramesPerSecond(float framesPerSecond);

        void setOutput(int output);

		void addParticles(Particle* particles, int nParticles);
		void addMassiveParticles(Particle* massiveParticles, int nMassiveParticles);
		void calcMassiveParticles(float timeElapsed, int step);

        void run();

        GUI *getGUI();

		v3 center;
		float radius;
};

#endif // CORE_H
