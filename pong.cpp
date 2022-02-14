#include <iostream>

#include "pong.h"

// Screen resolution
const int screen_width = 1024;
const int screen_height = 768;

// Wall thickness
const int thickness = 15;

// Paddle height
const int paddle_height = 100;

Pong::Pong()
{
	window_ = nullptr;
	is_running_ = true;
	is_done_ = false;

	// Reset to avoid undefined states
	reset();
}

Pong::~Pong()
{
}

bool Pong::initialize()
{
	// Initialize video system
	int sdl_result = SDL_Init(SDL_INIT_VIDEO);

	// Check for errors
	if (sdl_result != 0)
	{
		SDL_Log("Unable to initialize SDL: %s", SDL_GetError);

		return false;
	}

	window_ = SDL_CreateWindow(
		"Pong", // Title
		100,  // Top left x-coordinate of window
		100,  // Top left y-coordinate of window
		screen_width, // Width of window
		screen_height,  // Height of window
		0     // Flags (0 for no flags set)
	);

	// Check if window was created
	if (!window_)
	{
		SDL_Log("Unable to create window: %s", SDL_GetError);

		return false;
	}

	renderer_ = SDL_CreateRenderer(
		window_,
		-1,
		SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

	// Check if renderer was created
	if (!renderer_)
	{
		SDL_Log("Unable to create renderer: %s", SDL_GetError);

		return false;
	}

	reset();

	return true;
}

void Pong::run_loop()
{
	while (is_running_)
	{
		process_input();

		update_game();

		generate_output();
	}
}

void Pong::shutdown()
{
	// Free window
	SDL_DestroyWindow(window_);

	// Free renderer
	SDL_DestroyRenderer(renderer_);

	// Quit SDL system
	SDL_Quit();
}

Observation Pong::step(const std::vector<float>& actions, bool render)
{
	// 0 = None, 1 = Up, 2 = Down - Paddle 1
	paddle_dir_1 = 0;

	if (actions[0] == 1.0f)
		paddle_dir_1 = 1;

	if (actions[0] == 2.0f)
		paddle_dir_1 = -1;

	// 0 = None, 1 = Up, 2 = Down - Paddle 2
	paddle_dir_2 = 0;

	if (actions[1] == 1.0f)
		paddle_dir_2 = 1;

	if (actions[1] == 2.0f)
		paddle_dir_2 = -1;

	if (render)
	{
		update_game();

		generate_output();
	}
	else
	{
		// Simulate 60 FPS -> 0.016 s
		update_state(0.016f);
	}

	obs_.states = states_;
	obs_.done = is_done_;
	obs_.rewards = rewards_;

	return obs_;
}

void Pong::process_input()
{
	SDL_Event event;

	while (SDL_PollEvent(&event))
	{
		switch (event.type)
		{
		case SDL_QUIT:
			is_running_ = false;
			break;
		}
	}

	// Get entire keyboard state
	const Uint8* state = SDL_GetKeyboardState(nullptr);

	// If ESCAPE is pressed quit
	if (state[SDL_SCANCODE_ESCAPE])
	{
		is_running_ = false;
	}

	// If R is pressed reset game state
	if (state[SDL_SCANCODE_R])
	{
		reset();
	}

	// Update paddle direction 1
	paddle_dir_1 = 0;

	if (state[SDL_SCANCODE_S])
	{
		paddle_dir_1 += 1;
	}

	if (state[SDL_SCANCODE_W])
	{
		paddle_dir_1 -= 1;
	}

	// Update paddle direction 2
	paddle_dir_2 = 0;

	if (state[SDL_SCANCODE_K])
	{
		paddle_dir_2 += 1;
	}

	if (state[SDL_SCANCODE_I])
	{
		paddle_dir_2 -= 1;
	}

}

void Pong::update_game()
{
	// Wait until 16ms has elapsed since last frame
	while (!SDL_TICKS_PASSED(SDL_GetTicks(), ticks_count_ + 16));

	// Get delta since last time
	float delta_time = (SDL_GetTicks() - ticks_count_) / 1000.0f;

	// Clamp delta time to a maximum value to avoid pause effects
	if (delta_time > 0.05f)
	{
		delta_time = 0.05f;
	}

	// Update tick count
	ticks_count_ = SDL_GetTicks();


	update_state(delta_time);

}

void Pong::generate_output()
{
	// Set color on back buffer
	SDL_SetRenderDrawColor(renderer_, 0, 0, 255, 255);

	// Set back buffer to current color
	SDL_RenderClear(renderer_);

	// Change color for drawing objects
	SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 255);

	// Create a wall
	SDL_Rect wall{
		0, // Top left x
		0, // Top left y
		screen_width, // Width
		thickness // Height
	};

	// Render the top wall
	SDL_RenderFillRect(renderer_, &wall);

	// Create the bottom wall
	wall.y = screen_height - thickness;

	// Render the bottom wall
	SDL_RenderFillRect(renderer_, &wall);

	// Change color for drawing paddle and ball
	SDL_SetRenderDrawColor(renderer_, 227, 189, 18, 255);

	// Render ball
	SDL_Rect ball{
		static_cast<int>(ball_pos_.x - thickness / 2),
		static_cast<int>(ball_pos_.y - thickness / 2),
		thickness,
		thickness
	};

	// Render the ball
	SDL_RenderFillRect(renderer_, &ball);

	// Render the padding 1
	SDL_Rect padding_1{
		static_cast<int>(paddle_pos_1.x),
		static_cast<int>(paddle_pos_1.y - paddle_height / 2),
		thickness,
		paddle_height
	};

	// Render the paddle
	SDL_RenderFillRect(renderer_, &padding_1);

	// Render the padding 2
	SDL_Rect padding_2{
		static_cast<int>(paddle_pos_2.x),
		static_cast<int>(paddle_pos_2.y - paddle_height / 2),
		thickness,
		paddle_height
	};

	// Render the paddle
	SDL_RenderFillRect(renderer_, &padding_2);

	// Swap front and back buffer
	SDL_RenderPresent(renderer_);
}

void Pong::update_state(float delta_time)
{
	// Paddle Colision 1
	if (paddle_dir_1 != 0)
	{
		paddle_pos_1.y += paddle_dir_1 * 300.0f * delta_time;

		// Upper wall colision detection 
		if (paddle_pos_1.y > (screen_height - paddle_height / 2.0f - thickness))
		{
			paddle_pos_1.y = screen_height - paddle_height / 2.0f - thickness;
		}

		// Lower wall colision detection 
		if (paddle_pos_1.y < (paddle_height / 2.0f + thickness))
		{
			paddle_pos_1.y = paddle_height / 2.0f + thickness;
		}
	}

	// Paddle Colision2
	if (paddle_dir_2 != 0)
	{
		paddle_pos_2.y += paddle_dir_2 * 300.0f * delta_time;

		// Upper wall colision detection 
		if (paddle_pos_2.y > (screen_height - paddle_height / 2.0f - thickness))
		{
			paddle_pos_2.y = screen_height - paddle_height / 2.0f - thickness;
		}

		// Lower wall colision detection 
		if (paddle_pos_2.y < (paddle_height / 2.0f + thickness))
		{
			paddle_pos_2.y = paddle_height / 2.0f + thickness;
		}
	}

	// No reward
	rewards_[0] = 0.0f;
	rewards_[1] = 0.0f;

	// Update ball position
	ball_pos_.x += ball_vel_.x * delta_time;
	ball_pos_.y += ball_vel_.y * delta_time;

	float diff_1 = abs(paddle_pos_1.y - ball_pos_.y);
	float diff_2 = abs(paddle_pos_2.y - ball_pos_.y);

	// Paddle 1 and ball colision
	if (// Our y-difference is small enough
		diff_1 <= paddle_height / 2.0f &&
		// We are in the correct x-position
		ball_pos_.x <= 25.0f && ball_pos_.x >= 20.0f &&
		// The ball is moving to the left
		ball_vel_.x < 0.0f)
	{
		ball_vel_.x *= -1.2f; // Increase vel by 20%

		rewards_[0] = 1.0f;
	}
	// Paddle 2 and ball colision
	else if (// Our y-difference is small enough
		diff_2 <= paddle_height / 2.0f &&

		ball_pos_.x >= (screen_width - 25.0f) && ball_pos_.x <= (screen_width - 20.0f) &&
		// The ball is moving to the right
		ball_vel_.x > 0.0f)
	{
		ball_vel_.x *= -1.2f; // Increase vel by 20%

		rewards_[1] = 1.0f;
	}
	// Pong over P1
	else if (ball_pos_.x < 0.0f)
	{
		// P1 let the ball pass
		rewards_[0] = -10.0f;

		p2_score_ += 1;

		reset();
	}
	// Pong over P2
	else if (ball_pos_.x >= screen_width)
	{
		// P2 let the ball pass
		rewards_[1] = -10.0f;

		p1_score_ += 1;

		reset();
	}

	// Clamp ball velocity
	if (ball_vel_.x >= 450)
	{
		ball_vel_.x = 450;
	}

	if (ball_vel_.x <= -450)
	{
		ball_vel_.x = -450;
	}

	// Did the ball collide with the top wall?
	if (ball_pos_.y <= thickness && ball_vel_.y < 0.0f)
	{
		ball_vel_.y *= -1;
	}
	// Did the ball collide with the bottom wall?
	else if (ball_pos_.y >= (screen_height - thickness) &&
		ball_vel_.y > 0.0f)
	{
		ball_vel_.y *= -1;
	}

	// Agent 1 state

	// Update velocity
	states_[0][4] = (paddle_pos_1.x / screen_width - states_[0][0]) / delta_time;
	states_[0][5] = (paddle_pos_1.y / screen_height - states_[0][1]) / delta_time;
	states_[0][6] = (ball_pos_.x / screen_width - states_[0][2]) / delta_time;
	states_[0][7] = (ball_pos_.y / screen_height - states_[0][3]) / delta_time;

	// Update position
	states_[0][0] = paddle_pos_1.x / screen_width;
	states_[0][1] = paddle_pos_1.y / screen_height;
	states_[0][2] = ball_pos_.x / screen_width;
	states_[0][3] = ball_pos_.y / screen_height;

	// Agent 2 state

	// Update velocity
	states_[1][4] = (paddle_pos_2.x / screen_width - states_[1][0]) / delta_time;
	states_[1][5] = (paddle_pos_2.y / screen_height - states_[1][1]) / delta_time;
	states_[1][6] = (ball_pos_.x / screen_width - states_[1][2]) / delta_time;
	states_[1][7] = (ball_pos_.y / screen_height - states_[1][3]) / delta_time;

	// Update position
	states_[1][0] = paddle_pos_2.x / screen_width;
	states_[1][1] = paddle_pos_2.y / screen_height;
	states_[1][2] = ball_pos_.x / screen_width;
	states_[1][3] = ball_pos_.y / screen_height;
	
	

	// Episode ended
	if (p1_score_ >= max_score_ || p2_score_ >= max_score_)
	{
		is_done_ = true;

		p1_score_ = 0;
		p2_score_ = 0;
	}
}

std::vector<std::vector<float>> Pong::reset()
{
	// Initialize ball and paddle positions
	ball_pos_.x = screen_width / 2;
	ball_pos_.y = screen_height / 2.0f + (rand() % 200) - 100.0f;

	ball_vel_.x = -200.0f; // pixels/second on down

	if (rand() % 2 == 1)
	{
		ball_vel_.x = 200.0f; // pixels/second on down
	}
	
	ball_vel_.y = 235.0f;  // pixels/second on left

	if (rand() % 2 == 1)
	{
		ball_vel_.y = -235.0f;  // pixels/second on left
	}

	// Reset paddle 1 position
	paddle_pos_1.x = 10;
	paddle_pos_1.y = screen_height / 2;

	// Reset paddle 2 position
	paddle_pos_2.x = screen_width - thickness - 10;
	paddle_pos_2.y = screen_height / 2;

	// One state for each paddle agent
	states_.resize(2);

	states_[0].resize(8, 0.0f);
	states_[0][0] = paddle_pos_1.x / screen_width;
	states_[0][1] = paddle_pos_1.y / screen_height;
	states_[0][2] = ball_pos_.x / screen_width;
	states_[0][3] = ball_pos_.y / screen_height;

	states_[1].resize(8, 0.0f);
	states_[1][0] = paddle_pos_2.x / screen_width;
	states_[1][1] = paddle_pos_2.y / screen_height;
	states_[1][2] = ball_pos_.x / screen_width;
	states_[1][3] = ball_pos_.y / screen_height;

	// Reset rewards
	rewards_.resize(2, 0.0f);

	// Reset game over flag
	is_done_ = false;

	return states_;
}

Observation env_step(Pong & env, const std::vector<float> actions, int frame_skip, bool render)
{
	int i;

	std::vector<Observation> observations(frame_skip);

	for (i = 0; i < frame_skip; i++)
	{
		observations[i] = env.step(actions, render);
	}

	// Create states for two agents
	std::vector<std::vector<float>> next_states(2);
	next_states[0].resize(observations[0].states[0].size() * frame_skip);
	next_states[1].resize(observations[0].states[1].size() * frame_skip);

	// Initialize total reward
	std::vector<float> total_rewards(2, 0.0f);

	bool done = false;

	int count = 0;

	// For each frame
	for (i = 0; i < frame_skip; i++)
	{
		// For each observed state 
		for (size_t j = 0; j < observations[i].states[0].size(); j++)
		{
			// Update agent 1 state and normalize data
			next_states[0][count] = observations[i].states[0][j];

			// Update agent 2 state and normalize data
			next_states[1][count] = observations[i].states[1][j];

			count++;
		}

		// Update agent 1 reward
		total_rewards[0] += observations[i].rewards[0];

		// Update agent 2 reward
		total_rewards[1] += observations[i].rewards[1];

		if (observations[i].done) {
			done = true;
			break;
		}
	}

	return { next_states , total_rewards , done };
}
