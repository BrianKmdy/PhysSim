#include <cstdio>

#include "Paths.h"
#include "Types.h"
#include "Processor.h"

SDL_Window* m_pCompanionWindow;
SDL_GLContext m_pContext;
bool m_bVblank = false;
GLint m_nSceneMatrixLocation;
GLuint m_unSceneProgramID;
GLuint m_unCompanionWindowVAO;
GLuint m_unCompanionWindowProgramID;
GLuint m_unSceneVAO;
GLuint m_glSceneVertBuffer;
GLuint m_nPointMatrixLocation;

int width = 1500;
int height = 1000;

glm::vec4* points = nullptr;

YAML::Node gConfig;

std::vector<std::string> positionFiles;

int nParticles;

int frame = 0;

// ... other code below
bool Processor::init()
{
	spdlog::info("Loading configuration");
	if (std::filesystem::exists(OutputConfigFilePath) && std::filesystem::exists(PositionDataDirectory)) {
		gConfig = YAML::LoadFile(OutputConfigFilePath.string());
		nParticles = getNParticles(&gConfig);
		spdlog::info("nParticles: {}", nParticles);

		points = new glm::vec4[nParticles];
		memset(points, 0, nParticles * sizeof(glm::vec4));
		for (int i = 0; i < nParticles; i++)
			points[i].w = 1;

		for (auto& path : std::filesystem::directory_iterator(PositionDataDirectory)) {
			positionFiles.push_back(path.path().string());
		}

	}
	else {
		spdlog::error("Configuration file doesn't exist");

		return false;
	}

	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0)
	{
		printf("%s - SDL could not initialize! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	m_pCompanionWindow = SDL_CreateWindow("hellovr", 500, 200, width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
	if (m_pCompanionWindow == NULL)
	{
		printf("%s - Window could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	SDL_SetWindowFullscreen(m_pCompanionWindow, SDL_WINDOW_FULLSCREEN_DESKTOP);

	m_pContext = SDL_GL_CreateContext(m_pCompanionWindow);
	if (m_pContext == NULL)
	{
		printf("%s - OpenGL context could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	GLenum nGlewError = glewInit();
	if (nGlewError != GLEW_OK)
	{
		printf("%s - Error initializing GLEW! %s\n", __FUNCTION__, glewGetErrorString(nGlewError));
		return false;
	}

	if (SDL_GL_SetSwapInterval(m_bVblank ? 1 : 0) < 0)
	{
		printf("%s - Warning: Unable to set VSync! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	// Set up the shaders
	const char* name = "Scene";
	const char* vertexShader = "#version 410\n"
		"uniform mat4 matrix;\n"
		"layout(location = 0) in vec4 position;\n"
		"out vec4 v4Color;\n"
		"void main()\n"
		"{\n"
		"   v4Color = vec4(0, 1.0, 0, 1.0);\n"
		"	gl_Position = matrix * position;\n"
		"}\n";
	const char* fragmentShader = "#version 410\n"
		"in vec4 v4Color;\n"
		"out vec4 outputColor;\n"
		"void main()\n"
		"{\n"
		"   outputColor = v4Color;\n"
		"}\n";

	GLuint unProgramID = glCreateProgram();

	GLuint nSceneVertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(nSceneVertexShader, 1, &vertexShader, NULL);
	glCompileShader(nSceneVertexShader);

	GLint vShaderCompiled = GL_FALSE;
	glGetShaderiv(nSceneVertexShader, GL_COMPILE_STATUS, &vShaderCompiled);
	if (vShaderCompiled != GL_TRUE)
	{
		printf("%s - Unable to compile vertex shader %d!\n", name, nSceneVertexShader);
		glDeleteProgram(unProgramID);
		glDeleteShader(nSceneVertexShader);
		return 0;
	}
	glAttachShader(unProgramID, nSceneVertexShader);
	glDeleteShader(nSceneVertexShader); // the program hangs onto this once it's attached

	GLuint  nSceneFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(nSceneFragmentShader, 1, &fragmentShader, NULL);
	glCompileShader(nSceneFragmentShader);

	GLint fShaderCompiled = GL_FALSE;
	glGetShaderiv(nSceneFragmentShader, GL_COMPILE_STATUS, &fShaderCompiled);
	if (fShaderCompiled != GL_TRUE)
	{
		printf("%s - Unable to compile fragment shader %d!\n", name, nSceneFragmentShader);
		glDeleteProgram(unProgramID);
		glDeleteShader(nSceneFragmentShader);
		return 0;
	}

	glAttachShader(unProgramID, nSceneFragmentShader);
	glDeleteShader(nSceneFragmentShader); // the program hangs onto this once it's attached

	glLinkProgram(unProgramID);

	GLint programSuccess = GL_TRUE;
	glGetProgramiv(unProgramID, GL_LINK_STATUS, &programSuccess);
	if (programSuccess != GL_TRUE)
	{
		printf("%s - Error linking program %d!\n", name, unProgramID);
		glDeleteProgram(unProgramID);
		return 0;
	}

	glUseProgram(unProgramID);
	glUseProgram(0);

	m_nPointMatrixLocation = glGetUniformLocation(unProgramID, "matrix");
	if (m_nPointMatrixLocation == -1)
	{
		printf("Unable to find matrix uniform in render model shader\n");
		return false;
	}

	// Create the buffers
	glGenVertexArrays(1, &m_unSceneVAO);
	glBindVertexArray(m_unSceneVAO);

	glGenBuffers(1, &m_glSceneVertBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_glSceneVertBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * nParticles, points, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), 0);

	glBindVertexArray(0);
	glDisableVertexAttribArray(0);

	return true;
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
	while (alive && frame < positionFiles.size()) {
		// loadPosition();
		refresh();
		handleInput();

		frame++;
	}
}

void Processor::loadPosition()
{
	spdlog::info("Loading position file '{}'", positionFiles[frame]);

	std::ifstream positionFile(positionFiles[frame], std::ios_base::in | std::ios_base::binary);
	for (int i = 0; i < nParticles; i++) {
		positionFromFile(&positionFile, &points[i].x, &points[i].y);
	}
}

void Processor::refresh()
{
	glViewport(0, 0, width, height);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	points[0] = { 0, 0, 0, 1};
	points[1] = { 50, 100, 0, 1 };

	points[2] = { 0, 50, 0, 1 };
	points[3] = { 100, 100, 0, 1 };

	points[4] = { 50, 50, 0, 1 };
	points[5] = { 100, 100, 0, 1 };

	points[6] = { 100, 50, 0, 1 };
	points[7] = { 100, 100, 0, 1 };

	glBindVertexArray(m_unSceneVAO);
	glBindBuffer(GL_ARRAY_BUFFER, m_glSceneVertBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * nParticles, points, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, false, sizeof(glm::vec4), (GLvoid*) 0);
	glBindVertexArray(0);


	glUseProgram(m_unSceneProgramID);
	// glm::mat4 modelview;
	// glm::vec4 eye(0., 0., -60000);
	// glm::vec4 look(0., 0., 0.);
	// glm::vec4 up(1., 1., 0.);
	// modelview = glm::lookAt(eye, look, up);
	glm::mat4 view(1.0f);
	glUniformMatrix4fv(m_nPointMatrixLocation, 1, GL_FALSE, glm::value_ptr(view));
	glBindVertexArray(m_unSceneVAO);

	glDrawArrays(GL_LINES, 0, nParticles);

	glBindVertexArray(0);
	glUseProgram(0);
	SDL_GL_SwapWindow(m_pCompanionWindow);
}

void Processor::shutdown()
{
	if (m_pCompanionWindow)
	{
		SDL_DestroyWindow(m_pCompanionWindow);
		m_pCompanionWindow = NULL;
	}

	SDL_Quit();

	delete[] points;
}


