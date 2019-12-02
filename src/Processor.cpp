#include <cstdio>
#include <algorithm>

#include "Paths.h"
#include "Types.h"
#include "Processor.h"

#include "SDL.h"
#include "GL/glew.h"
#include "gl/gl.h"

SDL_Window* m_pCompanionWindow;
SDL_GLContext m_pContext;
bool m_bVblank = false;

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
	return bufferIndex < files.size() * stepSize;
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
		// As the frame index moves forward we can pop frames off to re-use them in the frame pool
		for (auto it = frames.begin(); it != frames.end();) {
			if (it->first + 2 * stepSize < frameIndex) {
				framePool.push_back(it->second);
				it = frames.erase(it);
			}
			else
			{
				break;
			}
		}

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

	spdlog::info("Framebuffer thread shutting down");
}

Processor::Processor():
	alive(true),
	frame(0),
	currentFrame(nullptr),
	nextFrame(nullptr),
	shader(nullptr),
	frameBuffer(nullptr),
	VBOCurrent(0),
    VBONext(0),
    VAO(0)
{
}

bool Processor::init(std::filesystem::path shaderPath, std::filesystem::path sceneDirectory)
{
	std::filesystem::path positionDirectory = sceneDirectory / PositionDirectoryName;
	std::filesystem::path configPath = sceneDirectory / ConfigFileName;

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

	// Start the frame buffer
	frameBuffer = std::make_shared<FrameBufferIn>(1000, nParticles, 25, positionFiles);
	frameBuffer->start();

	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0)
	{
		printf("%s - SDL could not initialize! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY );
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 0);

	m_pCompanionWindow = SDL_CreateWindow("Simulation", 0, 0, width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
	if (m_pCompanionWindow == NULL)
	{
		printf("%s - Window could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	// SDL_SetWindowFullscreen(m_pCompanionWindow, SDL_WINDOW_FULLSCREEN_DESKTOP);
	// SDL_GetWindowSize(m_pCompanionWindow, &width, &height);

	m_pContext = SDL_GL_CreateContext(m_pCompanionWindow);
	if (m_pContext == NULL)
	{
		printf("%s - OpenGL context could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	glewExperimental = GL_TRUE;
	GLenum nGlewError = glewInit();
	if (nGlewError != GLEW_OK)
	{
		printf("%s - Error initializing GLEW! %s\n", __FUNCTION__, glewGetErrorString(nGlewError));
		return false;
	}
	glGetError(); // to clear the error caused deep in GLEW

	glDebugMessageCallback((GLDEBUGPROC)DebugCallback, nullptr);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_DONT_CARE);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

	if (SDL_GL_SetSwapInterval(m_bVblank ? 1 : 0) < 0)
	{
		printf("%s - Warning: Unable to set VSync! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	spdlog::info("Loading shaders");

	// build and compile our shader program
	// ------------------------------------
	shader = std::make_shared<Shader>(shaderPath / "shader.vs", shaderPath / "shader.fs"); // you can name your shader files however you like

	glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBOCurrent);
	glGenBuffers(1, &VBONext);
	// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
	// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	// glBindVertexArray(0);
	glViewport(0, 0, width, height);

	// glEnable(GL_ALPHA_TEST);
	// glAlphaFunc(GL_EQUAL, 1.0);
}

void Processor::handleInput()
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

void Processor::run()
{
	spdlog::info("Running: {} positions", positionFiles.size());

	while (!frameBuffer->hasFramesBuffered()) {
		spdlog::info("Waiting for first frame");
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	// Get the first frame
	frameBuffer->nextFrame(&nextFrame);

	while (alive && (frameBuffer->hasFramesBuffered() || frameBuffer->hasMoreFrames())) {
		handleInput();

		if (frameBuffer->hasFramesBuffered() && frame >= frameBuffer->stepSize) {
			currentFrame = nextFrame;
			frameBuffer->nextFrame(&nextFrame);
			update();
			frame = 0;
		}

		if (currentFrame && nextFrame)
			refresh();

		frame++;
	}
}

void Processor::update()
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

void Processor::refresh()
{
	glClearColor(0.8f, 0.0f, 0.0f, 0.3f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	// Set the shader
	shader->use();
	shader->setFloat("time", std::min((float) (frame % frameBuffer->stepSize) / (float) frameBuffer->stepSize, 1.0f));

	// Draw the vertices
	glBindVertexArray(VAO);
	glPointSize(3.0f);
	glDrawArrays(GL_POINTS, 0, nParticles);

	// SwapWindow
	{
		SDL_GL_SwapWindow(m_pCompanionWindow);
	}
}

void Processor::shutdown()
{
	// optional: de-allocate all resources once they've outlived their purpose:
// ------------------------------------------------------------------------
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBOCurrent);
	glDeleteBuffers(1, &VBONext);

	if (m_pCompanionWindow)
	{
		SDL_DestroyWindow(m_pCompanionWindow);
		m_pCompanionWindow = NULL;
	}

	frameBuffer->stop();
	frameBuffer->join();

	SDL_Quit();
}