#include "agent.h"

DQNAgent::DQNAgent(
	int state_space,
	int action_space,
DQNAgentConfig config):
	state_space_(state_space),
	action_space_(action_space),
	config_(config),
	memory_(config.mem_size, config.batch_size)
{
	// Create online net 
	policy_net_.build({ state_space,config.hidden_fc1, config.hidden_fc2,action_space }, Activation::ReLU, Activation::None);

	policy_net_.set_lr(config.learning_rate);

	// Create target net
	target_net_.build({ state_space,config.hidden_fc1, config.hidden_fc2,action_space }, Activation::ReLU, Activation::None);

	// Initialize paramenters
	std::vector<float> policy_params = policy_net_.get_parameters();
	std::vector<float> target_params = target_net_.get_parameters();

	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(-3e-3, 3e-3);
	for (size_t i = 0; i < policy_params.size(); i++)
	{
		policy_params[i] = distribution(generator);
		target_params[i] = distribution(generator);
	}

	policy_net_.set_parameters(policy_params);
	target_net_.set_parameters(target_params);

}

DQNAgent::~DQNAgent()
{
}

float DQNAgent::get_action(const std::vector<float>& state, float epsilon)
{
	float action = 0.0f;

	// Exploratory action
	if (random_(engine_) < epsilon)
	{
		action = rand() % action_space_;

		return action;
	}
	// Policy action
	else
	{
		// Make a prediction based on current state
		std::vector<std::vector<float>> actions = policy_net_.predict({ state });

		int argmax_idx = -1;
		action = -std::numeric_limits<float>::max();

		// Take single result from batch
		for (size_t i = 0; i < actions[0].size(); i++)
		{
			if (actions[0][i] > action)
			{
				action = actions[0][i];
				argmax_idx = i;
			}
		}

		if (argmax_idx == -1)
			printf("Error");

		// Return the action of argmax(q,s)
		return static_cast<float>(argmax_idx);
	}

	return action;
}

void DQNAgent::step(std::vector<float>& state, float action, float reward, std::vector<float>& next_state, bool done)
{

	// Save experience in replay memory
	memory_.add_experience(state, action, reward, next_state, done);

	// Learn every UPDATE_EVERY time steps.
	iter_ = (iter_ + 1) % config_.update_every_iter;

	if (iter_ == 0)
	{
		// If enough samples are available in memory, get random subset and learn
		if (memory_.size() >= config_.batch_size)
		{
			Experiences experiences = memory_.sample();

			learn(experiences);
		}
	}
}

void DQNAgent::learn(const Experiences & experiences)
{
	// 1 - Get "index of maximum value" from actions predicted values for target model: Ex: [1,0,0,1,0]. Argmax(Q-values)
	std::vector<std::vector<float>> q_targets = policy_net_.predict(experiences.states);

	std::vector<std::vector<float>> q_expected = target_net_.predict(experiences.next_states);

	// Calculate Q Targets 
	for (size_t i = 0; i < config_.batch_size; i++)
	{
		int action_idx = static_cast<int>(experiences.actions[i]);

		q_targets[i][action_idx] = experiences.rewards[i];

		if (experiences.dones[i] == false)
		{
			int argmax_idx = -1;
			float argmax_q = -std::numeric_limits<float>::max();

			// Take single result from batch
			for (size_t j = 0; j < q_expected[i].size(); j++)
			{
				if (q_expected[i][j] > argmax_q)
				{
					argmax_q = q_expected[i][j];
					argmax_idx = j;
				}
			}

			q_targets[i][action_idx] += config_.gamma * argmax_q;
		}

	}

	float error = policy_net_.train_on_batch(experiences.states, q_targets, Loss::MSELoss, Optimizer::Adam);

	std::vector<float> policy_params = policy_net_.get_parameters();

	target_net_.set_parameters(policy_params);
}

void DQNAgent::save(const std::string & file_path)
{
	policy_net_.save(file_path);
}

void DQNAgent::load(const std::string & file_path)
{
	policy_net_.load(file_path);
}
