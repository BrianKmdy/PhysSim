#include "GUI.h"

#include <cstdlib>
#include <iostream>

double cameraAngle = 0;
double distance = 1000;

GUI::GUI(double wWidth, double wHeight) : UserInterface(wWidth, wHeight)
{
    if (!glfwInit())
        exit(EXIT_FAILURE);

    win = glfwCreateWindow(1920, 1080, "PhysSim", NULL, NULL);

//    particle.data = SOIL_load_image
//    (
//        "img/particle.tga",
//       &particle.width, &particle.height, &particle.channels,
//        SOIL_LOAD_RGBA
//    );

    //        glEnable(GL_LIGHTING);
    //glEnable(GL_DEPTH_TEST);
    //glEnable(GL_COLOR_MATERIAL);
    //glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    //glEnable(GL_LIGHT0);
    //glShadeModel(GL_FLAT);


    /*
    glEnable(GL_TEXTURE_2D);
    //glBindTexture(GL_TEXTURE_2D, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
    */

    //particle.id = SOIL_create_OGL_texture
    //(
    //    particle.data,
    //    particle.width, particle.height, particle.channels,
    //    SOIL_CREATE_NEW_ID,
    //    SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT
    //);

    //particle = SOIL_load_OGL_texture("img/particle.tga", 
    //                           SOIL_LOAD_AUTO, 
    //                           SOIL_CREATE_NEW_ID,
    //                           SOIL_FLAG_POWER_OF_TWO | 
    //                           SOIL_FLAG_MIPMAPS |
    //                           SOIL_FLAG_COMPRESS_TO_DXT);

    
    /*
    ilInit();
    ILuint image;// = LoadImage("img/particle.tga");
    ILboolean success;
    ilGenImages(1, &image);
    ilBindImage(image);
    success = ilLoadImage("img/particle.png");

    if (!win || !success) {
    //if (!win || !particle.data) {
        glfwTerminate();
        exit (EXIT_FAILURE);
    }

    success = ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE);

    if (!success) {
        glfwTerminate();
        exit (EXIT_FAILURE);
    }

    glGenTextures(1, &particle);
    glBindTexture(GL_TEXTURE_2D, particle); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
    glTexImage2D(GL_TEXTURE_2D, 0, ilGetInteger(IL_IMAGE_BPP), ilGetInteger(IL_IMAGE_WIDTH), ilGetInteger(IL_IMAGE_HEIGHT), 
    0, ilGetInteger(IL_IMAGE_FORMAT), GL_UNSIGNED_BYTE, ilGetData()); 

    //ilDeleteImages(1, &image);

    //std::cout<< sizeof(particle.data) << std::endl;

    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, particle.data);

    //std::cout << particle.width << std::endl;
    //std::cout<< (int)particle.data[0] << std::endl;
    //std::cout<< (int)particle.data[1] << std::endl;
    //std::cout<< (int)particle.data[2] << std::endl;
    //std::cout<< (int)particle.data[3] << std::endl;
    //std::cout<< (int)particle.data[4] << std::endl;
    //std::cout<< (int)particle.data[3000] << std::endl;
    //std::cout<< (int)particle.data[6] << std::endl;
    //std::cout<< (int)particle.data[7] << std::endl;
    */

    glfwMakeContextCurrent(win);
}

void GUI::setOutput(int output) {
    this->output = output;

    if (output == OUTPUT_TO_VIDEO)
        //outputVideo = popen("ffmpeg -y -f rawvideo -s 1800x1000 -pix_fmt rgb24 -r 30 -i - -vf vflip -an -b:v 10000k test.mp4", "w");
        outputVideo = popen("ffmpeg -y -f rawvideo -s 1920x1080 -pix_fmt rgb24 -r 30 -i - -vcodec libx264 -vf vflip -an test1.mp4", "w");
        //outputVideo.open("/home/brian/Desktop/video.avi", CV_FOURCC('D', 'I', 'V', 'X'), 30.0f, cv::Size(1280, 720), true);
}

void GUI::setCamera(Vector camera, Vector focus, Vector up) {
    this->camera = camera;
    this->focus = focus;
    this->up = up;
}

void GUI::tick(std::vector<Particle> * entities) {
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
    
    glEnable(GL_BLEND);
    glEnable(GL_ALPHA_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexEnvf(GL_TEXTURE_2D,GL_TEXTURE_ENV_MODE,GL_DECAL);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, particle.data);
    glEnable(GL_TEXTURE_2D);
    //glBindTexture(GL_TEXTURE_2D, particle);
    
    //glColor3f(0.0, 0.0, 0.0);
    for (std::vector<Particle>::iterator it = entities->begin(); it != entities->end(); it++) {
        /*
        float modelview[16];
        int i,j;

        // save the current modelview matrix
        glPushMatrix();

        // get the current modelview matrix
        glGetFloatv(GL_MODELVIEW_MATRIX , modelview);

        // undo all rotations
        // beware all scaling is lost as well 
        for( i=0; i<3; i++ ) 
            for( j=0; j<3; j++ ) {
                if ( i==j )
                    modelview[i*4+j] = 1.0;
                else
                    modelview[i*4+j] = 0.0;
            }

        // set the modelview with no rotations and scaling
        glLoadMatrixf(modelview);
        */

        //glRotatef(180, 1, 0, 0);
        glPushMatrix();
        glTranslatef(it->getX(), it->getY(), it->getZ());
        glColor3f(it->getR(), it->getG(), it->getB());
        //glRotatef(thetax, 1, 0, 0); 
        //glRotatef(thetay, 0, -1, 0);
        //glRotatef(thetaz, 0, 0, 1);
        glRotatef(theta, 0, 0, 1);
        glRotatef(phi, 0, 1, 0);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-it->getRadius(), -it->getRadius(), 0.f);
        glTexCoord2f(1.0, 0.0);
        glVertex3f(it->getRadius(), -it->getRadius(), 0.f);
        glTexCoord2f(1.0, 1.0);
        glVertex3f(it->getRadius(), it->getRadius(), 0.f);
        glTexCoord2f(0.0, 1.0);
        glVertex3f(-it->getRadius(), it->getRadius(), 0.f);
        glEnd();
        glPopMatrix();
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

//    outputVideo.release();
    if (outputVideo)
        pclose(outputVideo);
}

bool GUI::returnPressed() {
    if (glfwGetKey(win, GLFW_KEY_ENTER) == GLFW_PRESS)
        return true;
    else
        return false;
}
