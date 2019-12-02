#include <thread>

#include "cxxopts.hpp"

#include "Paths.h"
#include "Types.h"
#include "Processor.h"

int main(int argc, char* argv[])
{
	std::filesystem::path sceneDirectory = OutputDirectory;

	// Parse command line arguments
	try {
		cxxopts::Options options(argv[0], "Post processing and visualization of particle physics simulations");
		options
			.allow_unrecognised_options()
			.add_options()
			("help", "Print help")
			("s, show", "Play back the simulation in real time")
			("scene", "Path of the scene to load", cxxopts::value<std::string>());

		options.parse_positional({ "scene" });

		auto result = options.parse(argc, argv);

		if (result.count("help")) {
			std::cout << options.help() << std::endl;
			exit(0);
		}

		if (result.count("scene"))
			sceneDirectory = std::filesystem::path(result["scene"].as<std::string>());
	}
	catch (const cxxopts::OptionException& e) {
		spdlog::error("Error parsing arguments: {}", e.what());
		exit(1);
	}

	// Run the post processor
	Processor processor;
	if (processor.init(std::filesystem::path(argv[0]).remove_filename(), sceneDirectory)) {
		processor.run();
		processor.shutdown();
	}

	return 0;
}