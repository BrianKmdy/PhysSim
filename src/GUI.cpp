#include "GUI.h"

#include <cstdlib>

GUI::GUI(double wWidth, double wHeight) : UserInterface(wWidth, wHeight)
{
    if (!glfwInit())
        exit(EXIT_FAILURE);

    win = glfwCreateWindow(1280, 720, "PhysSim", NULL, NULL);

    if (!win) {
        glfwTerminate();
        exit (EXIT_FAILURE);
    }

    glfwMakeContextCurrent(win);
}

void GUI::setOutput(int output) {
    this->output = output;

    if (output == OUTPUT_TO_VIDEO)
        outputVideo.open("/home/brian/Desktop/video.avi", CV_FOURCC('D', 'I', 'V', 'X'), 30.0f, cv::Size(1280, 720), true);
}

void GUI::tick(const std::vector<Particle> * entities) {
    const double diameter = 0.003f;

    float ratio;
    int width, height;

    glfwGetFramebufferSize(win, &width, &height);
    ratio = width / (float) height;

    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
    glMatrixMode(GL_MODELVIEW);

    glBegin(GL_QUADS);
    for (std::vector<Particle>::const_iterator it = entities->begin(); it < entities->end(); it++) {
        glVertex3f(getX(it->getX()) - diameter, getY(it->getY()) - diameter, 0.f);
        glVertex3f(getX(it->getX()) + diameter, getY(it->getY()) - diameter, 0.f);
        glVertex3f(getX(it->getX()) + diameter, getY(it->getY()) + diameter, 0.f);
        glVertex3f(getX(it->getX()) - diameter, getY(it->getY()) + diameter, 0.f);
    }
    glEnd();

    glfwSwapBuffers(win);
    glfwPollEvents();

    if (output == OUTPUT_TO_VIDEO) {
        cv::Mat pixels( height, width, CV_8UC3 );
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data );
        cv::Mat cv_pixels( height, width, CV_8UC3 );
        for( int y=0; y<height; y++ ) for( int x=0; x<width; x++ ) {
            cv_pixels.at<cv::Vec3b>(y,x)[2] = pixels.at<cv::Vec3b>(height-y-1,x)[0];
            cv_pixels.at<cv::Vec3b>(y,x)[1] = pixels.at<cv::Vec3b>(height-y-1,x)[1];
            cv_pixels.at<cv::Vec3b>(y,x)[0] = pixels.at<cv::Vec3b>(height-y-1,x)[2];
        }
        outputVideo << cv_pixels;
    }
}

void GUI::drawParticle(Particle particle) {
    // TODO
}

bool GUI::shouldClose() {
    return glfwWindowShouldClose(win);
}

void GUI::terminate() {
    glfwDestroyWindow(win);
    glfwTerminate();

    outputVideo.release();
}

double GUI::getX(double x) {
    return (x / (wWidth / 2.f)) - 1.f;
}

double GUI::getY(double y) {
    return (y / (wHeight / 2.f)) - 1.f;
}

bool GUI::returnPressed() {
    if (glfwGetKey(win, GLFW_KEY_ENTER) == GLFW_PRESS)
        return true;
    else
        return false;
}
