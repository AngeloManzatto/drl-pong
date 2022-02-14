#pragma once

#include <SDL.h>
#include <vector>
#include <random>

struct Vector2
{
	float x;
	float y;
};

struct Observation
{
	std::vector<std::vector<float>> states;
	std::vector<float> rewards;
	bool done;
};

class Pong
{
public:
	Pong();
	~Pong();

	// Initialize the game
	bool initialize();

	// Run main game loop
	void run_loop();

	// Shut down the game
	void shutdown();

	// Single game step
	Observation step(const std::vector<float>& actions, bool render=false);

	std::vector<std::vector<float>> reset();

private:

	void process_input();

	void update_game();

	void generate_output();

	void update_state(float delta_time);

private:

	// Game window
	SDL_Window * window_;

	// Graph renderer
	SDL_Renderer * renderer_;

	// Monitor if game is running
	bool is_running_;

	// Monitor for game over
	bool is_done_;

	// Monitor reward
	std::vector<float> rewards_;

	// Track paddle position
	Vector2 paddle_pos_1;
	Vector2 paddle_pos_2;

	// Track ball position and velocity 
	Vector2 ball_pos_;
	Vector2 ball_vel_;

	// Game ticks
	Uint32 ticks_count_;

	// Paddle direction
	int paddle_dir_1;
	int paddle_dir_2;

	// Reinforcement Learning 
	std::vector<std::vector<float>> states_;

	// Game observations
	Observation obs_;

	int max_score_{ 21 };
	int p1_score_{ 0 };
	int p2_score_{ 0 };

};

Observation env_step(Pong& env, const std::vector<float> actions, int frame_skip, bool render = false);