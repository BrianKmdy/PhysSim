#include "argparse.hpp"
#include "spdlog/spdlog.h"

#include "Paths.h"
#include "Processor.h"

int getNParticles()
{

}

int main(int argc, const char** argv)
{
	argparse::ArgumentParser parser;

	parser.addArgument("-b");
	parser.addArgument("-c", "--cactus", 1);
	parser.addArgument("-o", "--optional");
	parser.addArgument("-r", "--required", 1, true);

	parser.parse(argc, argv);


	return 0;
}