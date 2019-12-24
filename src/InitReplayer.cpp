#include <thread>

#include "cxxopts.hpp"

#include "Paths.h"
#include "Types.h"
#include "Replayer.h"

int main(int argc, char* argv[])
{
	int frameStep = 1;
	float speed = 1.0;
	bool outputToVideo = false;
	std::filesystem::path sceneDirectory = OutputDirectory;

	// Parse command line arguments
	try {
		cxxopts::Options options(argv[0], "Post processing and visualization of particle physics simulations");
		options
			.add_options()
			("help", "Print help")
			("f, frameStep", "The step size between disk frames", cxxopts::value<int>())
			("s, speed", "Playback speed", cxxopts::value<float>())
			("v, video", "Output to video", cxxopts::value<bool>())
			("scene", "Path of the scene to load", cxxopts::value<std::string>());

		options.parse_positional({ "scene" });

		auto result = options.parse(argc, argv);

		if (result.count("help")) {
			std::cout << options.help() << std::endl;
			exit(0);
		}

		if (result.count("frameStep"))
			frameStep = result["frameStep"].as<int>();

		if (result.count("speed"))
			speed = result["speed"].as<float>();

		if (result.count("video"))
			outputToVideo = true;

		if (speed > frameStep) {
			spdlog::error("Playback speed must be less than or equal to frame step");
			exit(1);
		}

		if (result.count("scene"))
			sceneDirectory = std::filesystem::path(result["scene"].as<std::string>());
	}
	catch (const cxxopts::OptionException& e) {
		spdlog::error("Error parsing arguments: {}", e.what());
		std::this_thread::sleep_for(std::chrono::seconds(5));
		exit(1);
	}

	// Run the post replayer
	Replayer replayer(frameStep, speed, outputToVideo);
	if (replayer.init(std::filesystem::path(argv[0]).remove_filename(), sceneDirectory)) {
		replayer.run();
		replayer.shutdown();
	}

	return 0;
}