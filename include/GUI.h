#ifndef GUI_H
#define GUI_H

#include "Particle.h"
#include "UserInterface.h"

#include <GLFW/glfw3.h>
#include <SOIL.h>
//#include <stdio.h>
#include <IL/il.h>
#include <GL/glut.h>

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

        int output;

        Sprite particle;
        //GLuint particle;

        Vector camera = Vector();
        Vector focus = Vector();
        Vector up = Vector();

    public:
        static const int OUTPUT_TO_VIDEO  = 0;
        static const int OUTPUT_TO_SCREEN = 1;

        GUI(double wWidth, double wHeight);

        void setCamera(Vector camera, Vector focus, Vector up);
        void setOutput(int output);
        void tick(const std::vector<Particle> * entities);
        void drawParticle(Particle particle);
        bool shouldClose();
        void terminate();
        bool returnPressed();
};

#endif // GUI_H
