#include "Types.h"
#include "Paths.h"

const unsigned long long MAX_BUFFER_MEMORY = 10ULL * 1000ULL * 1000ULL * 1000ULL;

std::map <std::string, int> Kernel::fromString = {
	   {"gravity", Kernel::gravity},
	   {"experimental", Kernel::experimental}
};

std::map<int, std::string> Kernel::toString = {
	{Kernel::gravity, "gravity"},
	{Kernel::experimental, "experimental"}
};

std::chrono::milliseconds getMilliseconds()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
}

int getNParticles(YAML::Node* config)
{
	int nParticles = 0;

	YAML::Node particles = (*config)["particles"];
	if (particles.IsDefined()) {
		for (auto it = particles.begin(); it != particles.end(); ++it) {
			YAML::Node n = (*it)["n"];
			if (n.IsDefined()) {
				nParticles += n.as<float>();
			}
			else {
				nParticles++;
			}
		}
	}

	return nParticles;
}

void positionToFile(std::ofstream* file, float* x, float* y)
{
	file->write(reinterpret_cast<char*>(x), sizeof(float));
	file->write(reinterpret_cast<char*>(y), sizeof(float));
}

void positionFromFile(std::ifstream* file, float* x, float* y)
{
	file->read(reinterpret_cast<char*>(x), sizeof(float));
	file->read(reinterpret_cast<char*>(y), sizeof(float));
}

void particleToFile(std::ofstream* file, float* x, float* y, float* vx, float* vy, float* mass)
{
	file->write(reinterpret_cast<char*>(x), sizeof(float));
	file->write(reinterpret_cast<char*>(y), sizeof(float));
	file->write(reinterpret_cast<char*>(vx), sizeof(float));
	file->write(reinterpret_cast<char*>(vy), sizeof(float));
	file->write(reinterpret_cast<char*>(mass), sizeof(float));
}

void particleFromFile(std::ifstream* file, float* x, float* y, float* vx, float* vy, float* mass)
{
	file->read(reinterpret_cast<char*>(x), sizeof(float));
	file->read(reinterpret_cast<char*>(y), sizeof(float));
	file->read(reinterpret_cast<char*>(vx), sizeof(float));
	file->read(reinterpret_cast<char*>(vy), sizeof(float));
	file->read(reinterpret_cast<char*>(mass), sizeof(float));
}