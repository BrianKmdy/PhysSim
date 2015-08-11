#ifndef GUI_H
#define GUI_H

#include "Particle.h"
#include "UserInterface.h"

#include <GLFW/glfw3.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

class GUI : public UserInterface
{
    private:
        GLFWwindow *win;

        cv::VideoWriter outputVideo;

        int output;

        double getX(double x);
        double getY(double y);

    public:
        static const int OUTPUT_TO_VIDEO  = 0;
        static const int OUTPUT_TO_SCREEN = 1;

        GUI(double wWidth, double wHeight);

        void setOutput(int output);
        void tick(const std::vector<Particle> * entities);
        void drawParticle(Particle particle);
        bool shouldClose();
        void terminate();
        bool returnPressed();
};

#endif // GUI_H
