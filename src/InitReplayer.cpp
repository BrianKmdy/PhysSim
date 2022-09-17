#include <thread>

#include "cxxopts.hpp"

#include "Paths.h"
#include "Types.h"
#include "Replayer.h"

int main(int argc, char* argv[])
{
	Replayer replayer;
	replayer.setShaderPath(std::filesystem::path(argv[0]).remove_filename());

	// Parse command line arguments
	try {
		cxxopts::Options options(argv[0], "Post processing and visualization of particle physics simulations");
		options
			.add_options()
			("help", "Print help")
			("f, frameStep", "The step size between disk frames", cxxopts::value<int>())
			("s, speed", "Playback speed", cxxopts::value<float>())
			("r, radius", "Particle radius", cxxopts::value<float>())
			("v, video", "Output to video", cxxopts::value<bool>())
			("codec", "Video codec", cxxopts::value<std::string>())
			("format", "Video file format", cxxopts::value<std::string>())
			("scene", "Path of the scene to load", cxxopts::value<std::string>());

		options.parse_positional({ "scene" });

		auto result = options.parse(argc, argv);

		if (result.count("help")) {
			std::cout << options.help() << std::endl;
			exit(0);
		}

		if (result.count("frameStep"))
			replayer.setFrameStep(result["frameStep"].as<int>());

		if (result.count("speed"))
			replayer.setSpeed(result["speed"].as<float>());

		if (result.count("radius"))
			replayer.setParticleRadius(result["radius"].as<float>());

		if (result.count("video"))
			replayer.setOutputToVideo(true);

		if (result.count("codec"))
			replayer.setCodec(result["codec"].as<std::string>());

		if (result.count("format"))
			replayer.setFormat(result["format"].as<std::string>());

		if (result.count("scene"))
			replayer.setSceneDirectory(std::filesystem::path(result["scene"].as<std::string>()));
	}
	catch (const cxxopts::OptionException& e) {
		spdlog::error("Error parsing arguments: {}", e.what());
		std::this_thread::sleep_for(std::chrono::seconds(5));
		exit(1);
	}

	// Run the post replayer
	if (replayer.init()) {
		replayer.run();
		replayer.shutdown();
	}

	return 0;
}