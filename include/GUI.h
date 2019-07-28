#ifndef GUI_H
#define GUI_H

#include "Particle.h"
#include "UserInterface.h"

#include <cstdio>
#include <chrono>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glut.h>
#include <SOIL.h>

struct Sprite {
    unsigned char *data;
    int width;
    int height;
    int channels;
};

class GUI : public UserInterface
{
private:
	GLFWwindow *win;

	FILE *outputVideo = NULL;
	HANDLE hPipe;
	DWORD dwWritten;
	STARTUPINFO si;
	PROCESS_INFORMATION pi;

	std::string fileName;


	int output;

	Sprite particle;
	//GLuint particle;

	Vector camera = Vector();
	Vector focus = Vector();
	Vector up = Vector();

	std::chrono::milliseconds lastFrameTime;
	int lastFrameCount;

public:
	static const int OUTPUT_TO_VIDEO  = 0;
	static const int OUTPUT_TO_SCREEN = 1;

	GUI(double wWidth, double wHeight);
	virtual ~GUI() {}

	void setCamera(Vector camera, Vector focus, Vector up);
	void setOutput(int output, unsigned int fps);
	void setFileName(std::string fileName);
	void tick(Particle* particles, int nParticles);
	void drawParticle(Particle particle);
	bool shouldClose();
	void terminate();
	bool returnPressed();
};

#endif // GUI_H
