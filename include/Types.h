#pragma once

#include <fstream>
#include <map>

#include "yaml-cpp/yaml.h"

struct Kernel
{
	enum
	{
		unknown = 0,
		gravity,
		experimental
	};

	// static int fromString(std::string kernel);
	// static std::string toString(int kernel);

	static std::map<std::string, int> fromString;
	static std::map<int, std::string> toString;
};

int getNParticles(YAML::Node* config);

void positionToFile(std::ofstream* file, float* x, float* y);
void positionFromFile(std::ifstream* file, float* x, float* y);

void particleToFile(std::ofstream* file, float* x, float* y, float* vx, float* vy, float* mass);
void particleFromFile(std::ifstream* file, float* x, float* y, float* vx, float* vy, float* mass);