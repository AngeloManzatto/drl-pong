#include <iostream>

#include "pong.h"
#include "agent.h"

const int TRAIN = false;

void train(Pong& env, DQNAgent& player_1, DQNAgent& player_2, int n_episodes)
{

	int time_step = 0;

	float epsilon = 1.00f;
	float epsilon_min = 0.01f;
	float epsilon_decay = 0.995f;

	float max_score = -999999.0f;

	// Combined scores
	std::vector<float> p1_score;
	std::vector<float> p2_score;

	for (int episode = 1; episode <= n_episodes; episode++)
	{
		env.reset();

		// Agent actions
		std::vector<float> actions(2);

		// Env states
		std::vector<std::vector<float>> states(2);
		states[0].resize(24);
		states[1].resize(24);

		// Agent scores
		std::vector<float> scores(2);

		while (true)
		{
			actions[0] = player_1.get_action(states[0], epsilon);
			actions[1] = player_2.get_action(states[1], epsilon);

			auto obs = env_step(env, actions, 3);

			// Memorize experiences and learn
			player_1.step(states[0], actions[0], obs.rewards[0], obs.states[0], obs.done);
 			player_2.step(states[1], actions[1], obs.rewards[1], obs.states[1], obs.done);

			// Update scores
			scores[0] += obs.rewards[0];
			scores[1] += obs.rewards[1];

			states = obs.states;

			if (obs.done)
			{
				break;
			}
		}

		// Update total score
		p1_score.push_back(scores[0]);
		p2_score.push_back(scores[1]);

		float curr_score = std::max(scores[0], scores[1]);

		if (curr_score > max_score)
		{
			max_score = curr_score;

			std::cout << "Episode:" << episode << ",Score:" << max_score << ",Eps:" << epsilon << "\n";

			player_1.save("p1-weights.txt");
			player_2.save("p2-weights.txt");
		}

		// Update exploration rate
		if (epsilon > epsilon_min)
		{
			epsilon *= epsilon_decay;
		}


	}
}

void play(Pong& env, DQNAgent& player_1, DQNAgent& player_2, int n_episodes)
{

	for (int episode = 1; episode <= n_episodes; episode++)
	{
		env.reset();

		// Agent actions
		std::vector<float> actions(2);

		// Env states
		std::vector<std::vector<float>> states(2);
		states[0].resize(24);
		states[1].resize(24);

		// Agent scores
		std::vector<float> scores(2);

		while (true)
		{
			actions[0] = player_1.get_action(states[0]);
			actions[1] = player_2.get_action(states[1]);

			auto obs = env_step(env, actions, 3, true);

			// Update scores
			scores[0] += obs.rewards[0];
			scores[1] += obs.rewards[1];

			states = obs.states;

			if (obs.done)
			{
				break;
			}
		}
	}

}

int main(int argc, char * args[])
{

	Pong env;

	DQNAgent player_1(24, 3);
	DQNAgent player_2(24, 3);

	player_1.load("p1-weights.txt");
	player_2.load("p2-weights.txt");

	if (TRAIN)
	{
		train(env, player_1, player_2, 10000);
	}
	else
	{
		bool success = env.initialize();

		play(env, player_1, player_2, 10000);
	}

	return 0;
}