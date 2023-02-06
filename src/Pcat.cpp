#include "Paths.h"
#include "Types.h"
#include "Replayer.h"

#include <thread>
#include <cstdint>
#include "cxxopts.hpp"
#include "yaml-cpp/yaml.h"

struct particle {
    float x;
    float y;
    float vx;
    float vy;
    float mass;

    void toYAML(YAML::Emitter* out) {
        *out << YAML::BeginMap;
        *out << YAML::Key << "x";
        *out << YAML::Value << x;
        *out << YAML::Key << "y";
        *out << YAML::Value << y;
        *out << YAML::Key << "vx";
        *out << YAML::Value << vx;
        *out << YAML::Key << "vy";
        *out << YAML::Value << vy;
        *out << YAML::Key << "mass";
        *out << YAML::Value << mass;
        *out << YAML::EndMap;
    }
};

class ParticleCat {
    public:
        ParticleCat(std::string path):
            m_path(path) {}

        void printParticles() {
            // Load m_path as an ifstream
            std::ifstream particleFile;
            try {
                particleFile.open(m_path, std::ios::binary);
            }
            catch (std::exception& e) {
                std::cerr << "Error opening file: " << e.what() << std::endl;
                return;
            }

            try {
                particleFile.seekg (0, particleFile.end);
                uint64_t length = particleFile.tellg();
                particleFile.seekg (0, particleFile.beg);

                YAML::Emitter out;
                out << YAML::BeginSeq;

                uint64_t index = 0;
                while (index < length) {
                    particle p;
                    particleFromFile(&particleFile, &p.x, &p.y, &p.vx, &p.vy, &p.mass);
                    p.toYAML(&out);

                    index += sizeof(particle);
                }

                out << YAML::EndSeq;
                std::cout << out.c_str() << std::endl;

            }
            catch (std::exception& e) {
                std::cerr << "Error reading particles: " << e.what() << std::endl;
                return;
            }
        }

    private:
        std::filesystem::path m_path;
};


int main(int argc, char* argv[])
{
	// Parse command line arguments
	try {
		cxxopts::Options options(argv[0], "Post processing and visualization of particle physics simulations");
		options
			.add_options()
			("help", "Print help")
			("path", "Path of the position file to load", cxxopts::value<std::string>());

		options.parse_positional({ "path" });

		auto result = options.parse(argc, argv);

		if (result.count("help")) {
			std::cout << options.help() << std::endl;
			exit(0);
		}

		if (result.count("path")) {
			auto cat = ParticleCat(result["path"].as<std::string>());
            cat.printParticles();
        } else {
            std::cerr << "No path specified" << std::endl;
            exit(1);
        }
	}
	catch (const cxxopts::OptionException& e) {
		std::cerr << "Error parsing arguments: " << e.what() << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(5));
		exit(1);
	}

	return 0;
}