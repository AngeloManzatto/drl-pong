#include <algorithm>

#include "replay_buffer.h"

ReplayBuffer::ReplayBuffer(size_t max_size, size_t batch_size)
{

	max_size_ = max_size;
	batch_size_ = batch_size <= max_size ? batch_size : max_size;

	states_.resize(max_size_);
	actions_.resize(max_size_);
	rewards_.resize(max_size_);
	dones_.resize(max_size_);
	next_states_.resize(max_size_);

}
void ReplayBuffer::add_experience(
	const std::vector<float>& state,
	float action,
	float reward,
	const std::vector<float>& next_state,
	bool done
)
{
	states_[idx_] = state;
	actions_[idx_] = action;
	rewards_[idx_] = reward;
	next_states_[idx_] = next_state;
	dones_[idx_] = done;

	// Update and rotate index if reached end
	idx_ += 1;
	idx_ = idx_ % max_size_;

	size_ = size_ >= max_size_ ? max_size_ : size_ + 1;
}

ReplayBuffer::~ReplayBuffer()
{
}

Experiences ReplayBuffer::sample()
{
	Experiences experiences;

	size_t total_size = std::min(batch_size_, size_);

	// Define indices for sampling the replay memory
	batch_indices_.resize(size_);

	// Populate indices
	for (size_t i = 0; i < size_; i++)
	{
		batch_indices_[i] = size_ - i - 1;
	}

	// Shuffle indices samples
	std::random_shuffle(std::begin(batch_indices_), std::end(batch_indices_));

	// (s, a, r, s', done)
	experiences.states.resize(total_size);
	experiences.actions.resize(total_size);
	experiences.rewards.resize(total_size);
	experiences.next_states.resize(total_size);
	experiences.dones.resize(total_size);

	for (size_t i = 0; i < total_size; i++)
	{
		experiences.states[i] = states_[batch_indices_[i]];
		experiences.actions[i] = actions_[batch_indices_[i]];
		experiences.rewards[i] = rewards_[batch_indices_[i]];
		experiences.next_states[i] = next_states_[batch_indices_[i]];
		experiences.dones[i] = dones_[batch_indices_[i]];
	}

	return experiences;

};