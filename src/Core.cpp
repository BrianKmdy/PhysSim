#include <map>

#include "spdlog/spdlog.h"

#include "Core.h"

Core::Core():
	Core(nullptr)
{
}

Core::Core(Instance* instance):
	alive(true),
	instance(instance)
{
}

Core::~Core()
{
	if (instance)
		delete[] reinterpret_cast<char*>(instance);
}

Instance* Core::getInstance()
{
	return instance;
}

void Core::setInstance(Instance* instance)
{
	if (this->instance)
		delete[] reinterpret_cast<char*>(this->instance);

	this->instance = instance;
}

void Core::verifyConfiguration()
{
	if (!instance)
		throw std::exception("No instance set");

	spdlog::info("---");
	spdlog::info("Config settings");

	std::map<std::string, std::string> values;
	std::map<std::string, std::vector<std::string>> errors;

	// Check the dimensions
	values["dimensions"] = std::to_string(instance->dimensions);
	if (instance->dimensions & (instance->dimensions - 1))
		errors["dimensions"].push_back("must be a power of 2");
	
	// Check the divisions
	values["divisions"] = std::to_string(instance->divisions);
	if (instance->divisions & (instance->divisions - 1))
		errors["divisions"].push_back("must be a power of 2");

	for (auto& pair : values) {
		if (errors.find(pair.first) == errors.end()) {
			spdlog::info(pair.first + ": " + pair.second);
		}
		else {
			std::string errorString;
			for (auto& error : errors[pair.first]) {
				errorString += "(" + error + ")";
			}
			spdlog::error(pair.first + ": " + pair.second + " " + errorString);
		}
	}

	spdlog::info("---");

	if (!errors.empty())
		throw std::exception("Invalid configuration");
}

void Core::run() {
	verifyConfiguration();

	spdlog::info("Initialzing cuda");

	initialize(instance);

	spdlog::info("Running simulation");

	simulate(instance);

	spdlog::info("Shutting down");

	unInitialize();

    // while (true) {
    //     // Call cuda to think here
	// 	// Launch a kernel on the GPU with one thread for each element.
	// 	cudaDeviceSynchronize();
    // }


	// cudaFree(particles);
	// cudaFree(massiveParticles);
	// cudaDeviceReset();
}

void Core::kill() {
	alive = false;
}
