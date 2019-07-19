#include "GUI.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <cstdlib>
#include <iostream>
#include <iomanip>

double cameraAngle = 0;
double distance = 1000;

GUI::GUI(double wWidth, double wHeight) :
		UserInterface(wWidth, wHeight),
		output(0),
		lastFrameTime(0),
		lastFrameCount(0)
{
    if (!glfwInit())
        exit(EXIT_FAILURE);

	GLFWmonitor* monitor = glfwGetPrimaryMonitor();
	const GLFWvidmode* mode = glfwGetVideoMode(monitor);

	glfwWindowHint(GLFW_RED_BITS, mode->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

	win = glfwCreateWindow(mode->width, mode->height, "PhysSim", monitor, NULL);

    particle.data = SOIL_load_image
    (
        "img/particle.tga",
        &particle.width, &particle.height, &particle.channels,
        SOIL_LOAD_RGBA
    );

    // glEnable(GL_LIGHTING);
    // glEnable(GL_DEPTH_TEST);
    // glEnable(GL_COLOR_MATERIAL);
    // glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    // glEnable(GL_LIGHT0);
    // glShadeModel(GL_FLAT);

    glfwMakeContextCurrent(win);

    lastFrameTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
}

void GUI::setOutput(int output) {
    this->output = output;

	if (output == OUTPUT_TO_VIDEO)
	{
		// outputVideo = popen("ffmpeg -y -f rawvideo -s 1800x1000 -pix_fmt rgb24 -r 30 -i - -vf vflip -an -b:v 10000k test.mp4", "w");
		// outputVideo = popen("ffmpeg.exe -y -f rawvideo -s 1920x1080 -pix_fmt rgb24 -r 60 -i - -vcodec libx264 -vf vflip -an test.mp4", "w");
		// outputVideo.open("C:\\Users\\brian\\Desktop\\out.avi", CV_FOURCC('D', 'I', 'V', 'X'), 30.0f, cv::Size(1280, 720), true);
	}
}

void GUI::setCamera(Vector camera, Vector focus, Vector up) {
    this->camera = camera;
    this->focus = focus;
    this->up = up;
}

void GUI::tick(Particle* particles, int nParticles) {
    float ratio;
    int width, height;

    glfwGetFramebufferSize(win, &width, &height);
    ratio = width / (float) height;

    //glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT);

    //glMatrixMode(GL_PROJECTION);
    //glLoadIdentity();
    //glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
    //glMatrixMode(GL_MODELVIEW);

    //cameraAngle += 0.005;

    //camera.setComponents(0, distance * sin(cameraAngle), distance * cos(cameraAngle));
    //camera.setComponents(0, 0, distance);
    //focus.setComponents(0, 0, 0);

    
    //Vector lookVector = radVector.vProduct(Vector(1, 0, 0));

    glPushMatrix();
    //glTranslatef(5000, 0, 0); 
    gluPerspective(60, ratio, 1, 5000);
    gluLookAt(camera.getX(), camera.getY(), camera.getZ(), focus.getX(), focus.getY(), focus.getZ(), up.getX(), up.getY(), up.getZ());
    //gluLookAt(camera.getX(), camera.getY(), camera.getZ(), focus.getX(), focus.getY(), focus.getZ(), 1, 0, 0);

    //float thetax = atan2(camera.getY() - focus.getY(), camera.getZ() - focus.getZ()) * (180.0 / M_PI) + 90;
    //float thetay = atan2(camera.getX() - focus.getX(), camera.getZ() - focus.getZ()) * (180.0 / M_PI) + 90;
    //float thetaz = atan2(camera.getY() - focus.getY(), camera.getX() - focus.getX()) * (180.0 / M_PI);

    Vector radVector = camera.difference(focus);
    float theta = atan2(radVector.getY(), radVector.getX()) * (180.0 / M_PI);
    float phi = acos(camera.getZ() / radVector.getMagnitude()) * (180.0 / M_PI);
    
    // glEnable(GL_BLEND);
    // glEnable(GL_ALPHA_TEST);
    // glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // glTexEnvf(GL_TEXTURE_2D,GL_TEXTURE_ENV_MODE,GL_DECAL);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, particle.data);
    // glEnable(GL_TEXTURE_2D);
    //glBindTexture(GL_TEXTURE_2D, particle);
    
    //glColor3f(0.0, 0.0, 0.0);
    for(int i = 0; i < nParticles; i++)
	{
        //glRotatef(180, 1, 0, 0);
        // glPushMatrix();
        // glTranslatef(it->getX(), it->getY(), it->getZ());
        // glColor3f(it->getR(), it->getG(), it->getB());
        //glRotatef(thetax, 1, 0, 0); 
        //glRotatef(thetay, 0, -1, 0);
        //glRotatef(thetaz, 0, 0, 1);
        // glRotatef(theta, 0, 0, 1);
        // glRotatef(phi, 0, 1, 0);
        // glBegin(GL_QUADS);
        // glTexCoord2f(0.0, 0.0);
        // glVertex3f(-it->getRadius(), -it->getRadius(), 0.f);
        // glTexCoord2f(1.0, 0.0);
        // glVertex3f(it->getRadius(), -it->getRadius(), 0.f);
        // glTexCoord2f(1.0, 1.0);
        // glVertex3f(it->getRadius(), it->getRadius(), 0.f);
        // glTexCoord2f(0.0, 1.0);
        // glVertex3f(-it->getRadius(), it->getRadius(), 0.f);
        // glEnd();
        // glPopMatrix();

    	glBegin(GL_POINTS);
    	   glColor3f(particles[i].getR(), particles[i].getG(), particles[i].getB());
    	   glVertex3f(particles[i].getX(), particles[i].getY(), particles[i].getZ());
    	glEnd();
    }
    
    glDisable(GL_TEXTURE_2D);

    glPopMatrix();

    glfwSwapBuffers(win);
    glfwPollEvents();


    if (output == OUTPUT_TO_VIDEO) {
        //cv::Mat pixels( height, width, CV_8UC3 );
        //glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data );
        //cv::Mat cv_pixels( height, width, CV_8UC3 );
        //for( int y=0; y<height; y++ ) for( int x=0; x<width; x++ ) {
        //    cv_pixels.at<cv::Vec3b>(y,x)[2] = pixels.at<cv::Vec3b>(height-y-1,x)[0];
        //    cv_pixels.at<cv::Vec3b>(y,x)[1] = pixels.at<cv::Vec3b>(height-y-1,x)[1];
        //    cv_pixels.at<cv::Vec3b>(y,x)[0] = pixels.at<cv::Vec3b>(height-y-1,x)[2];
        //}
        //outputVideo << cv_pixels;
        int *pixels = new int[1920 * 1080 * 3];
        glReadPixels(0, 0, 1920, 1080, GL_RGB, GL_UNSIGNED_BYTE, pixels);
        if (outputVideo)
            fwrite(pixels, 1920*1080*3, 1, outputVideo);
        delete [] pixels;
    }
    else
    {
    	++lastFrameCount;

		std::chrono::milliseconds currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
		if(currentTime - lastFrameTime > std::chrono::seconds(1))
		{
			std::cout << "Fps: " << std::fixed << std::setprecision(2) << static_cast<float>(lastFrameCount) / (currentTime - lastFrameTime).count() * 1000.0 << std::endl;

			lastFrameTime = currentTime;
			lastFrameCount = 0;
		}
    }
}

void GUI::drawParticle(Particle particle) {
    // TODO
}

bool GUI::shouldClose() {
	if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS)
		return true;

    return glfwWindowShouldClose(win);
}

void GUI::terminate() {
    glfwDestroyWindow(win);
    glfwTerminate();

//    outputVideo.release();
	if (outputVideo)
	{
		// pclose(outputVideo);
	}
}

bool GUI::returnPressed() {
    if (glfwGetKey(win, GLFW_KEY_ENTER) == GLFW_PRESS)
        return true;
    else
        return false;
}

