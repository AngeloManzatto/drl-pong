#pragma once

#include <deque>
#include <vector>

struct Experiences
{
	std::vector<std::vector<float>> states;
	std::vector<float> actions;
	std::vector<float> rewards;
	std::vector<std::vector<float>> next_states;
	std::vector<bool> dones;
};

class ReplayBuffer
{
public:
	ReplayBuffer(size_t max_size, size_t batch_size);

	~ReplayBuffer();

	void add_experience(
		const std::vector<float>& state,
		float action,
		float reward,
		const std::vector<float>& next_state,
		bool done
	);

	Experiences sample();

	size_t size() { return size_; }
	size_t batch_size() { return batch_size_; }
	size_t max_size() { return max_size_; }

private:

	std::vector<std::vector<float>> states_;
	std::vector<float> actions_;
	std::vector<float> rewards_;
	std::vector<std::vector<float>> next_states_;
	std::vector<bool> dones_;

	size_t size_{ 0 };
	size_t max_size_;
	size_t batch_size_;

	size_t idx_{ 0 };

	std::vector<size_t> batch_indices_;
};