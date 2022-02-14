#pragma once

#include <vector>

namespace ann {

	/****************************************************************************************
	 * Neural Network Activations
	****************************************************************************************/
	enum class Activation
	{
		None,
		Sigmoid,
		Tanh,
		ReLU,
		LeakyReLU,
		Softmax
	};

	inline Activation str_to_activation(const std::string& activation)
	{
		if (strcmp(activation.c_str(), "sigmoid") == 0) return Activation::Sigmoid;
		if (strcmp(activation.c_str(), "tanh") == 0) return Activation::Tanh;
		if (strcmp(activation.c_str(), "relu") == 0) return Activation::ReLU;
		if (strcmp(activation.c_str(), "leaky_relu") == 0) return Activation::LeakyReLU;
		if (strcmp(activation.c_str(), "softmax") == 0) return Activation::Softmax;

		return Activation::None;
	}

	inline std::string activation_to_str(const Activation activation)
	{
		if (activation == Activation::Sigmoid) return "sigmoid";
		if (activation == Activation::Tanh) return "tanh";
		if (activation == Activation::ReLU) return "relu";
		if (activation == Activation::LeakyReLU) return "leaky_relu";
		if (activation == Activation::Softmax) return "softmax";

		return "none";
	}

	enum class Loss
	{
		None,
		CrossEntropyLoss,
		MSELoss
	};

	enum class Optimizer
	{
		None,
		SGD,
		RMSprop,
		Adam
	};

	class Net
	{
	public:

		Net() = default;

		~Net();

		Net(const std::vector<int>& layers,
			Activation hidden_activation = Activation::Sigmoid,
			Activation output_activation = Activation::Sigmoid
		);

		// Getters and Setters
		void set_lr(float lr) { lr_ = lr; }
		void set_rho(float rho) { rho_ = rho; }
		void set_eps(float eps) { eps_ = eps; }

		void build(
			const std::vector<int>& layers,
			Activation hidden_activation,
			Activation output_activation
		);

		void resize(const size_t batch);

		void set_parameters(const std::vector<float>& parameters);

		const std::vector<float>& get_parameters() const { return parameters_; }

		size_t w_offset(size_t layer_index);

		size_t b_offset(size_t layer_index);

		std::vector<std::vector<float>> predict(const std::vector<std::vector<float>>& inputs);

		void batch_to_vector(const std::vector<std::vector<float>>& b, std::vector<float>& v, size_t offset = 0);

		void vector_to_batch(const std::vector<float>& v, std::vector<std::vector<float>>& b, size_t offset = 0);

		void forward(const std::vector<std::vector<float>>& inputs);

		void backward(const std::vector<std::vector<float>>& costs);

		void activation_forward(
			const Activation activation,
			const int batch,
			const int n,
			const float * x, float * y);

		void activation_backward(
			const Activation activation,
			const int batch,
			const int n,
			const float * x, const float * y, const float * dy, float * dx);

		float calculate_loss(
			Loss loss,
			const std::vector<std::vector<float>>& outputs,
			const std::vector<std::vector<float>>& targets,
			std::vector<std::vector<float>>& costs
		);

		float train_on_batch(
			const std::vector<std::vector<float>>& X,
			const std::vector<std::vector<float>>& y,
			Loss loss,
			Optimizer optimizer);

		void train(
			const std::vector<std::vector<float>>& X,
			const std::vector<std::vector<float>>& y,
			size_t epochs,
			size_t batch,
			float learning_rate,
			Loss loss,
			Optimizer optimizer
		);

		std::vector<size_t> generate_indices(size_t size, bool random = true);


		void update_parameters(
			Optimizer optimizer,
			std::vector<float>& parameters,
			std::vector<float>& gradients
		);

		const std::vector<std::vector<float>>& outputs() const { return net_outputs_; }

		// Save and load network
		void save(const std::string& file_path);

		void load(const std::string& file_path);

		// Serialize and deserialize network
		void serialize(std::ostream& stream);

		void deserialize(std::istream& stream);

	private:
		std::vector<int> layers_;

		Activation hidden_activation_;
		Activation output_activation_;

		std::vector<std::vector<float>> hidden_;
		std::vector<std::vector<float>> costs_;

		std::vector<float> parameters_;
		std::vector<float> gradients_;

		std::vector<std::vector<float>> net_outputs_;
		std::vector<std::vector<float>> net_costs_;
		std::vector<std::vector<float>> net_inputs_;
		std::vector<std::vector<float>> net_targets_;

		size_t batch_;

		// Training parameters
		size_t iter_{ 1 };
		float net_error_{ 0.0f };
		float lr_{ 0.01f };
		float rho_{ 0.9f };
		float mom_{ 0.0f };
		float eps_{ 1e-7f };
		float beta_1_{ 0.9f };
		float beta_2_{ 0.999f };
		std::vector<float> v_;
		std::vector<float> m_;
		std::vector<float> errors_;

	};

}