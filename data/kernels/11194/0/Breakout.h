

#ifndef BREAKOUT_H
#define BREAKOUT_H

//#include "gl3w\gl3w.h"
//#include <GLFW\glfw3.h>
#include <irrklang\irrKlang.h>
#include <gl\glew.h>
#include <SDL\SDL.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <SOIL\SOIL.h>
#include <imgui\imgui.h>
#include <string>
#include <unordered_map>
#include <GLM/glm.hpp>
#include <GLM/gtc/matrix_transform.hpp>
#include "kernel.h"

#include <GLM\gtc\type_ptr.hpp>
#include <glm/gtx/vector_angle.hpp>

//#include <imgui\glut\imgui_impl_glut.h>
#include <imgui\imgui_impl_sdl_gl3.h>

using namespace glm;
using namespace std;
using namespace irrklang;
#if defined(__APPLE__) || defined(MACOSX)
	#include <GLUT/glut.h>
#else
	#include <gl\freeglut.h>
#endif

// My includes
#include "MyObjects.h"      // Game-specific objects
#include "config.h"         // Game configurations




enum class WindowFlag { WINDOWED, FULLSCREEN, EXCLUSIVE_FULLSCREEN, BORDERLESS };

namespace Engine {
class Breakout {

public:
	// Class Constructor/Destructor
	Breakout();
	~Breakout();

	// Public functions (handle GLUT calls)
	void PressKey(unsigned int keyID);
	void ReleaseKey(unsigned int keyID);
	void SetMouseCoords(float x, float y);
	/// Returns true if the key is held down
	bool IsKeyDown(unsigned int keyID);
	bool showgame;
	/// Returns true if the key was just pressed
	bool IsKeyPressed(unsigned int keyID);
	//getters
	vec2 GetMouseCoords() const { return _mouseCoords; }
	void buildBackgroundImage(void);
	void renderBackground(void);
	void drawGUI(void);
	void drawAfterGameGUI(char* gameresult, int score);
	void display(void);
	void init(void);
	void reshape(int width, int height);
	void mouseClick(int button, int state, int x, int y);
	void mouseMove(int x, int y);
	void keyStroke(unsigned char key, int x, int y);
	void specialKeyPos(int key, int x, int y);
	void startGame(string windowTitle, unsigned int screenWidth, unsigned int screenHeight, bool vsync, WindowFlag windowFlag, unsigned int targetFrameRate, float timeScale);
	void Engine::Breakout::buildBackground();

private:
	// Game statistics
	float targetOffset = 0.1f, cubeSpeed = 0.01f, rotation = 0;
	vec3 targetPosition, position, forward, axisRotation;
	int score;
	int level;
	int reward;
	int lifesCount;
	unsigned screenWidth, screenHeight;
	float timeScale;
	float targetFrame = 0;
	GLuint VAO, VBO, EBO, VAOBG, VBOBG, EBOBG ,program, program2, programBG, texture, textureBG;

	//background image
	SDL_Surface *back_surface;
	SDL_Texture *back_texture;
	SDL_Renderer *renderer;

	// Possible ame mode
	enum State { INIT, Menus, Gameplay, Scoreboard, EXIT };
	Breakout::State gameState;

	// Balls
	std::vector <Ball> balls;

	// Paddle
	Paddle paddle;

	// Bricks
	std::vector<Brick> bricks;

	// Private functions
	void OnSDLGameEvent(SDL_Event& evt);
	void PollInput();
	void drawBackground(void);
	void displayMenu(void);
	void drawGame(void);
	void newBall(float x, float y);
	void drawBalls(void);
	void initPaddle(void);
	void drawPaddle(void);
	void initBricks(void);
	void bricksLevel1(void);
	void bricksLevel2(void);
	void drawBricks(void);
	template <typename Iterator>
	int wallCollision(Iterator it);
	template <typename Iterator>
	bool brickCollision(Iterator it, Iterator br);
	template <typename Iterator>
	Iterator hitBrick(Iterator brick);
	void resetBricks();
	void drawLife(float x, float y);
	void drawGameStats(void);
	void drawScore(void);
	void drawCircle();
	void drawCoordinate(void);
	void checkShaderErrors(GLuint shader, string type);

	bool WasKeyDown(unsigned int keyID);
	unordered_map<unsigned int, bool> _keyMap;
	unordered_map<unsigned int, bool> _previousKeyMap;
	vec2 _mouseCoords;

protected:
	unsigned int lastFrame = 0, last = 0, _fps = 0, fps = 0;
	void UseShader(GLuint program);
	GLuint BuildShader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr);
	SDL_Window *window;
	SDL_Window *windowGUI;
	SDL_GLContext glContext;
	State state;
	void getFPS();
	void Err(string err);
	float getDeltaTime();
	int getFrameRate();
	void limitFPS();
	virtual void OnUserDefinedEvent(SDL_Event& evt) = 0 ;

};
}

#endif // BREAKOUT_H
