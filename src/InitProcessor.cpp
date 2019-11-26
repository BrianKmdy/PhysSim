#include <thread>

#include "cxxopts.hpp"
#include "spdlog/spdlog.h"

#include "Paths.h"
#include "Types.h"
#include "Processor.h"

int main(int argc, char* argv[])
{
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

	Processor processor;
	if (processor.init()) {
		auto start = getMilliseconds();
		while (getMilliseconds() - start < std::chrono::seconds(10))
			processor.refresh();

		processor.shutdown();
	}

	return 0;
}