#ifndef CORE_H
#define CORE_H

// I propose a new fundamental force as the possible source for gravity. I imagine that everywhere in space there is a fluid
// called the cosmic medium which is infinitely compressible and seeks to spread out as much as possible. What we recognize
// as gravity could simply be the absence of this force between bodies of mass. Note that this hypothesis is inspired by
// ideas such as dark energy and integrates the concept of an expanding universe
//
// Hypothesis of the universal medium
// 1. At the time of the big bang the cosmic medium was an infinitely compressed fluid
// 2. Between any infinitesimal division of the fluid there is a repulsive force causing it to spread apart
// 3. The fluid spread apart rapidly leading to areas of uniform density in the center of the universe and aread of
//    high density at the outer edges of the expanding universe
// 4. Over time as the density becomes uniform the fluid begins to condense into matter
// 5. Since matter is highly concentrated relative to the fluid its repulsive force is also much stronger
// 6. The highly condensed matter creates small bubbles or pockets around it with lower density fluid
// 7. Rather than gravity being an attractive force between matter it's actually just the lack of a repulsive force
//    as these pockets create low density areas of fluid between them
// 8. As the low density pockets of fluid form the matter is squeezed together by the fluid surrounding it
// 9. The fluid is also the medium through which electromagnic waves travel at the speed of light
// 10. Action at a distance is impossible (classical gravity), any force which acts over distance must travel through the medium
//
// Further testing needs to be done to prove the validity of this hypothesis
// ~Brian Kimball Moody

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

		void dumpToDisk(std::string filename, bool full = false);
		void loadFromDisk(std::string filename, bool full = false);

        void run();

        GUI *getGUI();

		v3 center;
		float radius;
};

#endif // CORE_H
