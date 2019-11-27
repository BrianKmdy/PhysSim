#include <cstdio>

#include "Paths.h"
#include "Types.h"
#include "Processor.h"

#include "SDL.h"
#include "GL/glew.h"
#include "gl/gl.h"

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

int width = 2560;
int height = 1440;

glm::vec3* points = nullptr;

YAML::Node gConfig;

std::vector<std::string> positionFiles;

int nParticles;

int frame = 0;

float worldSize = 131072.0f;
float renderWorldSize = 100.0f;

void APIENTRY DebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const char* message, const void* userParam)
{
	printf("GL Error: %s\n", message);
}

// XXX/bmoody Need to add handling of keyframes, preloading position data into a circular queue, and interpolating between keyframes
// XXX/bmoody Also need to add saving directly to images

Shader::Shader(const char* vertexPath, const char* fragmentPath)
{
	// 1. retrieve the vertex/fragment source code from filePath
	std::string vertexCode;
	std::string fragmentCode;
	std::ifstream vShaderFile;
	std::ifstream fShaderFile;
	// ensure ifstream objects can throw exceptions:
	vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try
	{
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
	catch (std::ifstream::failure e)
	{
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
	}
	const char* vShaderCode = vertexCode.c_str();
	const char* fShaderCode = fragmentCode.c_str();
	// 2. compile shaders
	unsigned int vertex, fragment;
	// vertex shader
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	checkCompileErrors(vertex, "VERTEX");
	// fragment Shader
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	checkCompileErrors(fragment, "FRAGMENT");
	// shader Program
	ID = glCreateProgram();
	glAttachShader(ID, vertex);
	glAttachShader(ID, fragment);
	glLinkProgram(ID);
	checkCompileErrors(ID, "PROGRAM");
	// delete the shaders as they're linked into our program now and no longer necessary
	glDeleteShader(vertex);
	glDeleteShader(fragment);
}
// activate the shader
// ------------------------------------------------------------------------
void Shader::use()
{
	glUseProgram(ID);
}
// utility uniform functions
// ------------------------------------------------------------------------
void Shader::setBool(const std::string& name, bool value) const
{
	glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
}
// ------------------------------------------------------------------------
void Shader::setInt(const std::string& name, int value) const
{
	glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}
// ------------------------------------------------------------------------
void Shader::setFloat(const std::string& name, float value) const
{
	glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::checkCompileErrors(unsigned int shader, std::string type)
{
	int success;
	char infoLog[1024];
	if (type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}

bool Processor::init()
{
	spdlog::info("Loading configuration");
	if (std::filesystem::exists(OutputConfigFilePath) && std::filesystem::exists(PositionDataDirectory)) {
		gConfig = YAML::LoadFile(OutputConfigFilePath.string());
		nParticles = getNParticles(&gConfig);
		spdlog::info("nParticles: {}", nParticles);

		points = new glm::vec3[nParticles];
		memset(points, 0, nParticles * sizeof(glm::vec3));

		for (auto& path : std::filesystem::directory_iterator(PositionDataDirectory)) {
			positionFiles.push_back(path.path().string());
		}

		std::sort(positionFiles.begin(), positionFiles.end(),
			[] (const std::string& a, const std::string& b) -> bool
			{
				int aN = std::stoi(a.substr(a.find('-') + 1, a.find('.')));
				int bN = std::stoi(b.substr(b.find('-') + 1, b.find('.')));
				return aN < bN;
			});
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

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	//SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY );
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 0);

	m_pCompanionWindow = SDL_CreateWindow("Simulation", 500, 200, width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
	if (m_pCompanionWindow == NULL)
	{
		printf("%s - Window could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	SDL_SetWindowFullscreen(m_pCompanionWindow, SDL_WINDOW_FULLSCREEN_DESKTOP);

	SDL_GetWindowSize(m_pCompanionWindow, &width, &height);

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
	ourShader = new Shader("shader.vs", "shader.fs"); // you can name your shader files however you like

	// set up vertex data (and buffer(s)) and configure vertex attributes
	// ------------------------------------------------------------------
//	float vertices[] = {
//		// positions         // colors
//		 10000.0f, -10000.0f, 0.0f,
//		-10000.0f, -0.5f, 0.0f,
//		 0.0f,  0.5f, 0.0f
//	};

	glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

	// Get a handle for the view matrix
	modelMatrix = glGetUniformLocation(ourShader->ID, "model");
	if (viewMatrix == -1)
	{
		printf("Unable to find matrix uniform in scene shader\n");
		return false;
	}

	// Get a handle for the view matrix
	viewMatrix = glGetUniformLocation(ourShader->ID, "view");
	if (viewMatrix == -1)
	{
		printf("Unable to find matrix uniform in scene shader\n");
		return false;
	}

	projectionMatrix = glGetUniformLocation(ourShader->ID, "projection");
	if (projectionMatrix == -1)
	{
		printf("Unable to find matrix uniform in scene shader\n");
		return false;
	}

	// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
	// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	// glBindVertexArray(0);
	glViewport(0, 0, width, height);

	alive = true;
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

	while (alive && frame < positionFiles.size()) {
		loadPosition();
		refresh();
		handleInput();
		frame++;
	}

	spdlog::info("Stopping: frame {} alive {}", frame, alive);
}

void Processor::loadPosition()
{
	std::ifstream positionFile(positionFiles[frame], std::ios_base::in | std::ios_base::binary);
	for (int i = 0; i < nParticles; i++) {
		positionFromFile(&positionFile, &points[i].x, &points[i].y);
	}
}

void Processor::refresh()
{
	// render
// ------

		// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, nParticles * sizeof(glm::vec3), points, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*) 0);
	glEnableVertexAttribArray(0);

	// Set up the view matrix
	glm::vec3 eye(0., 0., 1000.0f);
	glm::vec3 look(0., 0., 0.);
	glm::vec3 up(0, 1., 0.);

	glm::mat4 model = glm::scale(glm::vec3(renderWorldSize / worldSize));
	glm::mat4 view = glm::lookAt(eye, look, up);
	glm::mat4 projection = glm::perspective(glm::radians(180.0f), (float)width / (float)height, 10.0f, 10000.0f);
	
	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// render the triangle
	ourShader->use();
	
	// Set the matrices
	glUniformMatrix4fv(modelMatrix, 1, GL_FALSE, glm::value_ptr(model));
	glUniformMatrix4fv(viewMatrix, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(projectionMatrix, 1, GL_FALSE, glm::value_ptr(projection));

	// Draw the vertices
	glBindVertexArray(VAO);
	glPointSize(1.5f);
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
	glDeleteBuffers(1, &VBO);

	if (m_pCompanionWindow)
	{
		SDL_DestroyWindow(m_pCompanionWindow);
		m_pCompanionWindow = NULL;
	}

	SDL_Quit();

	delete[] points;
	delete ourShader;
}