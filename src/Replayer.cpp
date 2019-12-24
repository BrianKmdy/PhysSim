#include <cstdio>
#include <algorithm>
#include <fcntl.h>
#include <io.h>

#include "Paths.h"
#include "Types.h"
#include "Replayer.h"

#include "SDL.h"
#include "GL/glew.h"
#include "gl/gl.h"

SDL_Window* m_pCompanionWindow;
SDL_GLContext m_pContext;
bool m_bVblank = true;

int width = 2560;
int height = 1440;

glm::vec3* points = nullptr;

YAML::Node gConfig;

std::map<int, std::string> positionFiles;

int nParticles;

float worldSize = 131072.0f;
float renderWorldSize = 100.0f;

void APIENTRY DebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const char* message, const void* userParam)
{
	printf("GL Error: %s\n", message);
}

// XXX/bmoody Need to add handling of keyframes, preloading position data into a circular queue, and interpolating between keyframes
// XXX/bmoody Also need to add saving directly to images

Shader::Shader(std::filesystem::path vertexPath, std::filesystem::path fragmentPath)
{
	// 1. retrieve the vertex/fragment source code from filePath
	std::string vertexCode;
	std::string fragmentCode;
	std::ifstream vShaderFile;
	std::ifstream fShaderFile;

	// ensure ifstream objects can throw exceptions:
	vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try {
		// open files
		vShaderFile.open(vertexPath);
		fShaderFile.open(fragmentPath);
		std::stringstream vShaderStream, fShaderStream;
		// read file's buffer contents into streams
		vShaderStream << vShaderFile.rdbuf();
		fShaderStream << fShaderFile.rdbuf();
		// close file handlers
		vShaderFile.close();
		fShaderFile.close();
		// convert stream into string
		vertexCode = vShaderStream.str();
		fragmentCode = fShaderStream.str();
	}
	catch (std::ifstream::failure e) {
		spdlog::error("Unable to load shader file: {}", e.what());
	}

	const char* vShaderCode = vertexCode.c_str();
	const char* fShaderCode = fragmentCode.c_str();
	// 2. compile shaders
	unsigned int vertex, fragment;
	
	// vertex shader
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	checkCompileErrors(vertex, "Vertex");

	// fragment Shader
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	checkCompileErrors(fragment, "Fragment");

	// shader Program
	ID = glCreateProgram();
	glAttachShader(ID, vertex);
	glAttachShader(ID, fragment);
	glLinkProgram(ID);
	checkCompileErrors(ID, "Program");

	// delete the shaders as they're linked into our program now and no longer necessary
	glDeleteShader(vertex);
	glDeleteShader(fragment);
}

void Shader::use()
{
	glUseProgram(ID);
}

void Shader::setBool(const std::string& name, bool value) const
{
	glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
}

void Shader::setInt(const std::string& name, int value) const
{
	glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::setFloat(const std::string& name, float value) const
{
	glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::setMatrix(const std::string& name, glm::mat4 value) const
{
	glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(value));
}

void Shader::checkCompileErrors(unsigned int shader, std::string type)
{
	int success;
	char infoLog[1024];
	if (type != "Program")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			spdlog::error("Shader compiliation error (type {}): {}", type, infoLog);
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			spdlog::error("Opengl program linking error (type {}): {}", type, infoLog);
		}
	}
}

FrameBufferIn::FrameBufferIn(int queueSize, int nParticles, int stepSize, std::map<int, std::string> files):
	FrameBuffer<glm::vec3>(queueSize, nParticles, stepSize),
	stepSize(stepSize),
	files(files)
{
}

bool FrameBufferIn::hasMoreFrames()
{
	return bufferIndex <= files.rbegin()->first;
}

 void FrameBufferIn::nextFrame(std::shared_ptr<glm::vec3[]>* frame)
{
	*frame = nullptr;

	try {
		*frame = frames.at(frameIndex);
		frameIndex += stepSize;
	}
	catch (std::out_of_range e) {
		spdlog::error("Unable to get frame at index {}", frameIndex);
	}
}

void FrameBufferIn::run()
{
	spdlog::info("Framebuffer thread starting up");

	while (alive) {
		// XXX/bmoody Re-enable this when we reach a point where we can't buffer all the frames at once
		// As the frame index moves forward we can pop frames off to re-use them in the frame pool
		// if (framePool.empty()) {
		// 	for (auto it = frames.begin(); it != frames.end();) {
		// 		if (it->first + 2 * stepSize < frameIndex) {
		// 			framePool.push_back(it->second);
		// 			it = frames.erase(it);
		// 		}
		// 		else
		// 		{
		// 			break;
		// 		}
		// 	}
		// }

		// If we have frames available to load and space in the pool then load the next frame
		if (hasMoreFrames() && !framePool.empty()) {
			try {
				// spdlog::info("Loading file {}", files[bufferIndex]);
				std::ifstream positionFile(files[bufferIndex], std::ios_base::in | std::ios_base::binary);
				glm::vec3* positions = &framePool.back()[0];

				for (int i = 0; i < nParticles; i++) {
					positionFromFile(&positionFile, &positions[i].x, &positions[i].y);
				}
				positionFile.close();

				frames[bufferIndex] = framePool.back();
				framePool.pop_back();
				bufferIndex += stepSize;
			}
			catch (std::exception e) {
				spdlog::error("Unable to read frame {} from file", bufferIndex);
			}
		}
		// Otherwise sleep
		else {
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}
}

Replayer::Replayer(int frameStep, float speed, bool outputToVideo):
	alive(true),
	frameStep(frameStep),
	interFrames(frameStep / speed),
	bufferSize(0),
	frame(0),
	currentFrame(nullptr),
	nextFrame(nullptr),
	shader(nullptr),
	controls(nullptr),
	frameBuffer(nullptr),
	VBOCurrent(0),
    VBONext(0),
    VAO(0),
	VBOControls(0),
	VAOControls(0),
	glFrameBuffer(0),
	outputToVideo(outputToVideo),
	outputVideoPipe(NULL)
{
}

bool Replayer::init(std::filesystem::path shaderPath, std::filesystem::path sceneDirectory)
{
	std::filesystem::path positionDirectory = sceneDirectory / PositionDirectoryName;
	std::filesystem::path configPath = sceneDirectory / ConfigFileName;

	spdlog::info("Parameters");
	spdlog::info("frameStep: {}", frameStep);
	spdlog::info("interFrames: {}", interFrames);

	spdlog::info("Loading configuration");
	if (std::filesystem::exists(configPath) && std::filesystem::exists(positionDirectory)) {
		gConfig = YAML::LoadFile(configPath.string());
		nParticles = getNParticles(&gConfig);
		spdlog::info("nParticles: {}", nParticles);

		points = new glm::vec3[nParticles];
		memset(points, 0, nParticles * sizeof(glm::vec3));

		for (auto& path : std::filesystem::directory_iterator(positionDirectory)) {
			std::string stringPath = path.path().string();
			positionFiles[std::stoi(stringPath.substr(stringPath.find('-') + 1, stringPath.find('.')))] = stringPath;
		}
	}
	else {
		spdlog::error("Configuration file doesn't exist");

		return false;
	}

	// XXX/bmoody Get this from the config
	int fileStep = std::next(positionFiles.begin())->first - positionFiles.begin()->first;
	if (frameStep < fileStep) {
		spdlog::error("Frame step must be larger than file step size (frameStep: {}, fileStep: {})", frameStep, fileStep);
		return false;
	}

	if (frameStep % fileStep != 0) {
		spdlog::error("Frame step must be a multiple of file step size (frameStep: {}, fileStep: {})", frameStep, fileStep);
		return false;
	}

	// Add 1 for position file 0
	int bufferSize = positionFiles.size() / (frameStep / fileStep) + 1;
	unsigned long long bufferBytes = static_cast<unsigned long long>(bufferSize) * static_cast<unsigned long long>(nParticles) * sizeof(glm::vec3);

	spdlog::info("Buffer size: {}", bufferSize);
	spdlog::info("Buffer memory: {:.0f}mb", static_cast<float>(bufferBytes) / 1000000.0);

	if (bufferBytes > MAX_BUFFER_MEMORY) {
		spdlog::error("Too many position files, will require allocating {:.0f}mb", static_cast<float>(bufferBytes) / 1000000.0);
		return false;
	}

	// Start the frame buffer
	frameBuffer = std::make_shared<FrameBufferIn>(bufferSize, nParticles, frameStep, positionFiles);
	frameBuffer->start();

	while (frameBuffer->bufferSize() < bufferSize - 1) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		spdlog::info("Buffering: {:.0f}%", static_cast<float>(frameBuffer->bufferSize()) / static_cast<float>(bufferSize) * 100.0);
	}

	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0)
	{
		spdlog::error("{} - SDL could not initialize! SDL Error: {}", __FUNCTION__, SDL_GetError());
		return false;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY );
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 0);

	int flags = SDL_WINDOW_OPENGL;
	if (outputToVideo)
		flags |= SDL_WINDOW_HIDDEN;
	else
		flags |= SDL_WINDOW_SHOWN;
	m_pCompanionWindow = SDL_CreateWindow("Simulation", 0, 0, width, height, flags);
	if (m_pCompanionWindow == NULL)
	{
		spdlog::error("{} - Window could not be created! SDL Error: {}", __FUNCTION__, SDL_GetError());
		return false;
	}

	// SDL_SetWindowFullscreen(m_pCompanionWindow, SDL_WINDOW_FULLSCREEN_DESKTOP);
	// SDL_GetWindowSize(m_pCompanionWindow, &width, &height);

	m_pContext = SDL_GL_CreateContext(m_pCompanionWindow);
	if (m_pContext == NULL)
	{
		spdlog::error("{} - OpenGL context could not be created! SDL Error: {}", __FUNCTION__, SDL_GetError());
		return false;
	}

	glewExperimental = GL_TRUE;
	GLenum nGlewError = glewInit();
	if (nGlewError != GLEW_OK)
	{
		spdlog::error("{} - Error initializing GLEW! {}", __FUNCTION__, glewGetErrorString(nGlewError));
		return false;
	}
	glGetError(); // to clear the error caused deep in GLEW

	glDebugMessageCallback((GLDEBUGPROC)DebugCallback, nullptr);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_DONT_CARE);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

	if (SDL_GL_SetSwapInterval(m_bVblank ? 1 : 0) < 0)
	{
		spdlog::error("{} - Warning: Unable to set VSync! SDL Error: {}", __FUNCTION__, SDL_GetError());
		return false;
	}

	spdlog::info("Loading shaders");

	// build and compile our shader program
	// ------------------------------------
	shader = std::make_shared<Shader>(shaderPath / "shader.vs", shaderPath / "shader.fs"); // you can name your shader files however you like
	controls = std::make_shared<Shader>(shaderPath / "controls.vs", shaderPath / "controls.fs"); // you can name your shader files however you like

	// Generate the vertex array for the particles
	glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBOCurrent);
	glGenBuffers(1, &VBONext);

	// Generate the vertex array for the controls
	glGenVertexArrays(1, &VAOControls);
	glGenBuffers(1, &VBOControls);

	glm::vec3 progressBarData[2] = { {0, 0, 0}, {2, 0, 0} };
	glBindVertexArray(VAOControls);
	glBindBuffer(GL_ARRAY_BUFFER, VBOControls);
	glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(glm::vec3), progressBarData, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(0);

	glGenFramebuffers(1, &glFrameBuffer);

	// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
	// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	// glBindVertexArray(0);
	glViewport(0, 0, width, height);

	// glEnable(GL_ALPHA_TEST);
	// glAlphaFunc(GL_EQUAL, 1.0);

	if (outputToVideo) {
		outputVideoPipe = _popen(std::string("ffmpeg.exe -y -f rawvideo -s " +
								 std::to_string(width) + "x" + std::to_string(height) +
			                     " -pix_fmt rgb24 -r 60 -i - -vcodec libx264 -vf vflip -an latest.mp4").c_str(), "w");
		_setmode(_fileno(outputVideoPipe), _O_BINARY);
	}

	return true;
}

void Replayer::handleInput()
{
	SDL_Event event;
	while (SDL_PollEvent(&event) != 0) {
		if (event.type == SDL_QUIT) {
			alive = false;
		}
		else if (event.type == SDL_KEYDOWN) {
			if (event.key.keysym.sym == SDLK_ESCAPE || event.key.keysym.sym == SDLK_q) {
				alive = false;
			}
		}
	}
}

void Replayer::run()
{
	spdlog::info("Running: {} positions", positionFiles.size());

	while (!frameBuffer->hasFramesBuffered()) {
		spdlog::info("Waiting for first frame");
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	// Get the first frame
	frameBuffer->nextFrame(&nextFrame);

	while (alive) {
		handleInput();

		if (!frameBuffer->hasMoreFrames() && !frameBuffer->hasFramesBuffered()) {
			if (outputToVideo)
				break;

			reset();
		}

		if (frame % interFrames == 0) {
			if (frameBuffer->hasFramesBuffered()) {
				currentFrame = nextFrame;
				frameBuffer->nextFrame(&nextFrame);
				update();
				refresh();
				frame++;
			}
		}
		else {
			if (currentFrame && nextFrame) {
				refresh();
				frame++;
			}
		}
	}
}

void Replayer::reset()
{
	frame = 0;
	frameBuffer->reset();
	frameBuffer->nextFrame(&nextFrame);
}

void Replayer::update()
{
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBOCurrent);
	glBufferData(GL_ARRAY_BUFFER, nParticles * sizeof(glm::vec3), currentFrame.get(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, VBONext);
	glBufferData(GL_ARRAY_BUFFER, nParticles * sizeof(glm::vec3), nextFrame.get(), GL_STATIC_DRAW);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(1);

	// Set up the view matrix
	glm::vec3 eye(0., 0., 1000.0f);
	glm::vec3 look(0., 0., 0.);
	glm::vec3 up(0, 1., 0.);

	glm::mat4 model = glm::scale(glm::vec3(renderWorldSize / worldSize * 0.5));
	glm::mat4 view = glm::lookAt(eye, look, up);
	glm::mat4 projection = glm::perspective(glm::radians(180.0f), (float)width / (float)height, 10.0f, 10000.0f);

	// XXX/bmoody Don't have to set these every frame
	shader->setMatrix("model", model);
	shader->setMatrix("view", view);
	shader->setMatrix("projection", projection);
}

void Replayer::refresh()
{
	glClearColor(0.8f, 0.0f, 0.0f, 0.3f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	if (!outputToVideo) {
		// Set the shader for the controls and draw
		controls->use();
		controls->setFloat("time", static_cast<float>(frame) / static_cast<float>(frameBuffer->bufferSize() * interFrames));

		glBindVertexArray(VAOControls);
		glLineWidth(3.0f);
		glDrawArrays(GL_LINES, 0, 2);
	}

	// Set the shader for the particles and draw
	shader->use();
	shader->setFloat("time", static_cast<float>(frame % interFrames) / static_cast<float>(interFrames));

	glBindVertexArray(VAO);
	glPointSize(3.0f);
	glDrawArrays(GL_POINTS, 0, nParticles);

	// SwapWindow
	SDL_GL_SwapWindow(m_pCompanionWindow);

	// XXX/bmoody No need to re-allocate memory on every frame, can just use a pre-allocated buffer
	if (outputToVideo) {
		unsigned char* pixels = new unsigned char[static_cast<size_t>(width) * static_cast<size_t>(height) * 3];
		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
		if (outputVideoPipe)
			fwrite(pixels, 3 * sizeof(unsigned char), static_cast<size_t>(width) * static_cast<size_t>(height), outputVideoPipe);
		delete[] pixels;
	}
}

void Replayer::shutdown()
{
	// optional: de-allocate all resources once they've outlived their purpose:
// ------------------------------------------------------------------------
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBOCurrent);
	glDeleteBuffers(1, &VBONext);

	if (outputToVideo) {
		_pclose(outputVideoPipe);
	}

	if (m_pCompanionWindow) {
		SDL_DestroyWindow(m_pCompanionWindow);
		m_pCompanionWindow = NULL;
	}

	frameBuffer->stop();
	frameBuffer->join();

	SDL_Quit();
}