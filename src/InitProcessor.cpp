#include <thread>

#include "cxxopts.hpp"

#include "Paths.h"
#include "Types.h"
#include "Processor.h"

int main(int argc, char* argv[])
{
	// Parse command line arguments
	try {
		cxxopts::Options options(argv[0], "Post processing and visualization of particle physics simulations");
		options
			.add_options()
			("help", "Print help")
			("s, show", "Play back the simulation in real time");

		auto result = options.parse(argc, argv);

		if (result.count("help")) {
			std::cout << options.help() << std::endl;
			exit(0);
		}
	}
	catch (const cxxopts::OptionException& e) {
		std::cout << "Error parsing arguments: " << e.what() << std::endl;
		exit(1);
	}

	// Run the post processor
	Processor processor;
	if (processor.init(std::filesystem::path(argv[0]).remove_filename().string())) {
		processor.run();
		processor.shutdown();
	}

	return 0;
}