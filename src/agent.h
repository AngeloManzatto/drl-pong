#pragma once

#include <random>
#include <vector>

#include "net.h"
#include "replay_buffer.h"

using namespace ann;

struct DQNAgentConfig
{
	float learning_rate{ 0.0005f };
	float gamma{ 0.95f };
	
	int hidden_fc1{ 128 };
	int hidden_fc2{ 128 };

	size_t mem_size{ 10000 };
	size_t batch_size{ 64 };
	
	size_t update_every_iter{ 20 };
};

class DQNAgent
{
public:
	DQNAgent(
		int state_space, 
		int action_space,
		DQNAgentConfig config= DQNAgentConfig()
		);
	~DQNAgent();

	float get_action(const std::vector<float>& state, float epsilon = 0.0f);

	void step(
		std::vector<float>& state,
		float action,
		float reward,
		std::vector<float>& next_state,
		bool done);

	void learn(const Experiences& experiences);

	// Save agent NN model
	void save(const std::string& file_path);

	// Load agent NN model
	void load(const std::string& file_path);

private:
	int state_space_;
	int action_space_;
	
	Net policy_net_;
	Net target_net_;

	ReplayBuffer memory_;

	DQNAgentConfig config_;

	// Random engine
	std::random_device rd_;
	std::mt19937 engine_{ rd_() };
	std::uniform_real_distribution<float> random_{ 0.0f, 1.0f };

	//
	int iter_ = 0;
};