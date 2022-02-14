#include <iostream>
#include <random>
#include <map>
#include <sstream>
#include <fstream>
#include <chrono>

#include "net.h"
#include "ann.h"

ann::Net::~Net()
{

}

ann::Net::Net(const std::vector<int>& layers, Activation hidden_activation, Activation output_activation)
{
	build(layers, hidden_activation, output_activation);
}

void ann::Net::build(const std::vector<int>& layers, Activation hidden_activation, Activation output_activation)
{
	if (layers.size() < 2)
		throw "Shapes must be highter than 1";

	layers_ = layers;
	hidden_activation_ = hidden_activation;
	output_activation_ = output_activation;

	size_t i, n_inputs, n_outputs, n_parameters;

	// Count total number of parameters
	n_parameters = 0;
	for (i = 0; i < layers_.size() - 1; i++)
	{
		n_inputs = layers_[i];
		n_outputs = layers_[i + 1];
		n_parameters += n_inputs * n_outputs + n_outputs; // weights + bias
	}

	// Resize parameters and gradient
	parameters_.resize(n_parameters);
	gradients_.resize(n_parameters);

	// Initialize parameters
	for (i = 0; i < parameters_.size(); i++)
	{
		parameters_[i] = -1.0f + 2.0f * ((float)rand()) / RAND_MAX;
	}

	// Resize hidden and costs
	hidden_.resize((layers_.size() - 1) * 2 + 1);
	costs_.resize((layers_.size() - 1) * 2 + 1);

	// On case for two layers only we assume there is not hidden activation
	if (layers.size() == 2)
		hidden_activation_ = output_activation_;
}

void ann::Net::resize(const size_t batch)
{
	size_t i;

	if (batch_ == batch)
		return;

	batch_ = batch;

	// Resize hidden and cost cache
	hidden_[0].resize(layers_[0] * batch_);
	costs_[0].resize(layers_[0] * batch_);
	for (i = 0; i < layers_.size() - 1; i++)
	{
		// Resize by n_outputs * batch 
		hidden_[2 * i + 1].resize(layers_[i + 1] * batch_); // Fc Layer
		hidden_[2 * i + 2].resize(layers_[i + 1] * batch_); // Activation

		costs_[2 * i + 1].resize(layers_[i + 1] * batch_); // Fc Layer
		costs_[2 * i + 2].resize(layers_[i + 1] * batch_); // Activation
	}

	// Resize net outputs and net costs
	net_outputs_.resize(batch_);
	net_costs_.resize(batch_);
	for (i = 0; i < net_outputs_.size(); i++)
	{
		net_outputs_[i].resize(layers_.back());
		net_costs_[i].resize(layers_.back());
	}

}

void ann::Net::set_parameters(const std::vector<float>& parameters)
{
	if (parameters.size() != parameters_.size())
		throw "Number of parameters while loading does not match!";

	for (size_t i = 0; i < parameters_.size(); i++)
	{
		parameters_[i] = parameters[i];
	}
}

size_t ann::Net::w_offset(size_t layer_index)
{
	size_t offset = 0;

	for (size_t i = 0; i < layer_index; i++)
	{
		offset += layers_[i] * layers_[i + 1] + layers_[i + 1];
	}

	return offset;
}

size_t ann::Net::b_offset(size_t layer_index)
{
	size_t offset = layers_[0] * layers_[1];

	for (size_t i = 0; i < layer_index; i++)
	{
		offset += layers_[i + 1] * layers_[i + 2] + layers_[i + 1];
	}

	return offset;
}

std::vector<std::vector<float>> ann::Net::predict(const std::vector<std::vector<float>>& inputs)
{
	forward(inputs);

	return net_outputs_;
}

void ann::Net::batch_to_vector(const std::vector<std::vector<float>>& b, std::vector<float>& v, size_t offset)
{
	for (size_t i = 0; i < b.size(); i++)
	{
		memcpy(v.data() + offset, b[i].data(), b[i].size() * sizeof(float));
		offset += b[i].size();
	}
}

void ann::Net::vector_to_batch(const std::vector<float>& v, std::vector<std::vector<float>>& b, size_t offset)
{
	for (size_t i = 0; i < b.size(); i++)
	{
		memcpy(b[i].data(), v.data() + offset, b[i].size() * sizeof(float));
		offset += b[i].size();
	}
}

void ann::Net::forward(const std::vector<std::vector<float>>& inputs)
{
	size_t n_layers = layers_.size() - 1;

	// Resize cache according to batch size
	resize(inputs.size());

	// Define current activation
	Activation activation = layers_.size() == 2 ? output_activation_ : hidden_activation_;

	// Convert inputs into vectorized format
	batch_to_vector(inputs, hidden_[0]);

	size_t layer_i = 0;

	// 1st pass
	fc_layer_forward(
		batch_, layers_[0], layers_[1], //batch, n_in, n_out
		hidden_[layer_i].data(), // x
		parameters_.data() + w_offset(0), // w
		parameters_.data() + b_offset(0), // b
		hidden_[layer_i + 1].data() // y
	);

	layer_i++;

	activation_forward(
		activation,
		batch_, // batch
		layers_[1], // n_elements
		hidden_[layer_i].data(), // x
		hidden_[layer_i + 1].data()  // y
	);

	layer_i++;

	// 
	for (size_t i_l = 1; i_l < n_layers; i_l++)
	{
		fc_layer_forward(
			batch_, layers_[i_l], layers_[i_l + 1],
			hidden_[layer_i].data(), // Previous result from activation layer
			parameters_.data() + w_offset(i_l),
			parameters_.data() + b_offset(i_l),
			hidden_[layer_i + 1].data()
		);

		layer_i++;

		activation = i_l == n_layers - 1 ? output_activation_ : hidden_activation_;

		activation_forward(
			activation,
			batch_,
			layers_[i_l + 1],
			hidden_[layer_i].data(),
			hidden_[layer_i + 1].data());

		layer_i++;
	}

	// Convert outputs to vector of vectors
	vector_to_batch(hidden_.back(), net_outputs_);

}

void ann::Net::activation_forward(
	const Activation activation,
	const int batch,
	const int n,
	const float * x, float * y)
{
	switch (activation)
	{
	case ann::Activation::None:memcpy(y, x, batch * n * sizeof(float));
		break;
	case ann::Activation::Sigmoid:sigmoid_forward(batch * n, x, y);
		break;
	case ann::Activation::Tanh:tanh_forward(batch * n, x, y);
		break;
	case ann::Activation::ReLU:relu_forward(batch * n, x, y);
		break;
	case ann::Activation::Softmax:softmax_forward(x, n, 1, batch, y);
		break;
	default:
		break;
	}
}

void ann::Net::backward(const std::vector<std::vector<float>>& costs)
{
	size_t n_layers = layers_.size() - 1;

	Activation activation = n_layers == 2 ? output_activation_ : hidden_activation_;

	batch_to_vector(costs, costs_.back());

	size_t layer_i = n_layers * 2;

	// Output Layer
	activation_backward(
		output_activation_, batch_, layers_[n_layers],
		hidden_[layer_i - 1].data(), // x
		hidden_[layer_i].data(),     // y
		costs_[layer_i].data(),      // dy
		costs_[layer_i - 1].data()   // dx
	);

	layer_i--;

	fc_layer_backward(
		batch_, layers_[n_layers - 1], layers_[n_layers], //batch, n_in, n_out
		hidden_[layer_i - 1].data(),                 // x
		parameters_.data() + w_offset(n_layers - 1), // w
		costs_[layer_i].data(),                      // dy
		gradients_.data() + w_offset(n_layers - 1),  // dw
		gradients_.data() + b_offset(n_layers - 1),  // db
		costs_[layer_i - 1].data()                   // dx 
	);

	layer_i--;

	for (size_t i_l = n_layers - 1; i_l-- > 0;)
	{

		activation_backward(
			hidden_activation_, batch_, layers_[i_l + 1],
			hidden_[layer_i - 1].data(), // x
			hidden_[layer_i].data(),     // y
			costs_[layer_i].data(),      // dy
			costs_[layer_i - 1].data()   // dx
		);

		layer_i--;

		fc_layer_backward(
			batch_, layers_[i_l], layers_[i_l + 1], // batch, n_in, n_out
			hidden_[layer_i - 1].data(),        // x
			parameters_.data() + w_offset(i_l), // w
			costs_[layer_i].data(),             // dy
			gradients_.data() + w_offset(i_l),  // dw
			gradients_.data() + b_offset(i_l),  // db
			costs_[layer_i - 1].data()          // dx
		);

		layer_i--;
	}
}

void ann::Net::activation_backward(
	const Activation activation,
	const int batch,
	const int n,
	const float * x, const float * y, const float * dy, float * dx)
{
	switch (activation)
	{
	case ann::Activation::None:memcpy(dx, dy, batch * n * sizeof(float));
		break;
	case ann::Activation::Sigmoid:sigmoid_backward(batch * n, x, dy, dx);
		break;
	case ann::Activation::Tanh:tanh_backward(batch * n, x, dy, dx);
		break;
	case ann::Activation::ReLU:relu_backward(batch * n, x, dy, dx);
		break;
	case ann::Activation::Softmax:softmax_backward(y, n, 1, batch, dy, dx);
		break;
	default:
		break;
	}
}

float ann::Net::calculate_loss(
	Loss loss,
	const std::vector<std::vector<float>>& outputs,
	const std::vector<std::vector<float>>& targets,
	std::vector<std::vector<float>>& costs
)
{
	size_t i;
	float error = 0.0f;

	for (i = 0; i < outputs.size(); i++)
	{
		switch (loss)
		{
		case ann::Loss::None:
			break;

		case ann::Loss::CrossEntropyLoss:

			// Calculate error
			error += cross_entropy_loss_layer_forward(outputs[i].size(), outputs[i].data(), targets[i].data());

			// Calculate loss
			cross_entropy_loss_layer_backward(outputs[i].size(), outputs[i].data(), targets[i].data(), net_costs_[i].data());

			break;

		case ann::Loss::MSELoss:

			// Calculate error
			error += mse_loss_layer_forward(outputs[i].size(), outputs[i].data(), targets[i].data());

			// Calculate loss
			mse_loss_layer_backward(outputs[i].size(), outputs[i].data(), targets[i].data(), net_costs_[i].data());

			break;

		default:
			break;
		}
	}

	return error / outputs.size();

}

void ann::Net::update_parameters(
	Optimizer optimizer,
	std::vector<float>& parameters,
	std::vector<float>& gradients
)
{
	// Apply clipping to avoid exploding gradients
	for (size_t i = 0; i < gradients.size(); i++)
	{
		if (gradients[i] > 10000)
		{
			gradients[i] = 10000;
		}

		if (gradients[i] < -10000)
		{
			gradients[i] = -10000;
		}
	}

	// Resize auxiliary optimizer gradients
	if (v_.size() != gradients.size())
	{
		v_.resize(gradients.size());
		m_.resize(gradients.size());
	}

	switch (optimizer)
	{
	case ann::Optimizer::SGD:sgd_optimizer(
		parameters.size(), lr_, mom_,
		gradients.data(), v_.data(), parameters.data());
		break;
	case ann::Optimizer::RMSprop:rms_prop_optimizer(
		parameters.size(), lr_, rho_, mom_, eps_,
		gradients.data(), v_.data(), parameters.data());
		break;
	case ann::Optimizer::Adam:adam_optimizer(
		parameters.size(), iter_, lr_, beta_1_, beta_2_, eps_,
		gradients.data(), v_.data(), m_.data(), parameters.data());
		break;
	default:
		break;
	}

	iter_ += 1;
}

std::vector<size_t> ann::Net::generate_indices(size_t size, bool random)
{
	std::vector<size_t> indices(size);

	// Populate indices
	for (size_t i = 0; i < indices.size(); i++)
	{
		indices[i] = indices.size() - i - 1;
	}

	if (random)
	{
		// Shuffle indices
		std::random_shuffle(std::begin(indices), std::end(indices));
	}

	return indices;
}

void ann::Net::train(
	const std::vector<std::vector<float>>& X,
	const std::vector<std::vector<float>>& y,
	size_t epochs,
	size_t batch,
	float learning_rate,
	Loss loss,
	Optimizer optimizer
)
{
	if (X.size() != y.size())
		throw "X and y sizes are different for training";

	size_t batch_size;
	auto start = std::chrono::high_resolution_clock::now();

	lr_ = learning_rate;

	// Initialize error history
	errors_.resize(epochs);

	// Resize net_inputs
	net_inputs_.resize(batch);
	net_targets_.resize(batch);

	std::vector<size_t> indices = generate_indices(X.size());

	// Epoch step

	for (size_t e_i = 0; e_i < epochs; e_i++)
	{
		// Batch step
		net_error_ = 0.0f;
		batch_size = int(X.size() / batch);
		for (int b_i = 0; b_i < batch_size; b_i++)
		{

			// FIX!!! Put inside method
			for (int i = 0; i < batch; i++)
			{
				// If we run out of indices, generate new ones
				if (indices.size() <= 0)
				{
					indices = generate_indices(X.size());
				}

				// Generate batch
				size_t idx = indices.back();
				indices.pop_back(); // Remove index

				net_inputs_[i] = X[idx];
				net_targets_[i] = y[idx];
			}

			net_error_ += train_on_batch(net_inputs_, net_targets_, loss, optimizer);
		}

		std::cout << "Error: " << net_error_ / batch_size << " Iter: " << iter_ << "\n";

		// Fix accumulate
		errors_[e_i] = net_error_;
	}

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

float ann::Net::train_on_batch(
	const std::vector<std::vector<float>>& X,
	const std::vector<std::vector<float>>& y,
	Loss loss,
	Optimizer optimizer)
{
	// Reset Gradients
	memset(gradients_.data(), 0, gradients_.size() * sizeof(float));

	// Forward Pass
	forward(X);

	// Error and Cost
	float error = calculate_loss(loss, net_outputs_, y, net_costs_);

	// Backward Pass
	backward(net_costs_);

	// Update parameters
	update_parameters(optimizer, parameters_, gradients_);

	return error;

}

void ann::Net::save(const std::string& file_path)
{
	std::ofstream file;

	file.open(file_path);

	serialize(file);

	file.close();
}

void ann::Net::load(const std::string& file_path)
{
	std::ifstream file;

	file.open(file_path);

	deserialize(file);

	file.close();
}

void ann::Net::serialize(std::ostream& stream)
{
	stream << "layers=[";

	for (size_t i = 0; i < layers_.size() - 1; i++)
	{
		stream << layers_[i] << " ";
	}
	stream << layers_.back() << "]";

	stream << "\n\n";

	stream << "activations=[";

	stream << activation_to_str(hidden_activation_) << " " << activation_to_str(output_activation_) << "]\n\n";

	stream << "parameters=[";
	for (size_t i = 0; i < parameters_.size() - 1; i++)
	{
		stream << parameters_[i] << " ";
	}
	stream << parameters_.back() << "]";
}

void ann::Net::deserialize(std::istream& stream)
{
	std::map<std::string, std::string> net_data;

	std::string line;

	// Extract net params from string
	while (std::getline(stream, line))
	{
		std::istringstream iss(line);
		unsigned first = line.find("[");
		unsigned last = line.find("]");

		if (line.find("layers") == 0)
		{

			std::string params = line.substr(first + 1, last - first - 1);
			net_data.insert({ "layers", params });
		}

		if (line.find("activations") == 0)
		{

			std::string params = line.substr(first + 1, last - first - 1);
			net_data.insert({ "activations", params });

		}

		if (line.find("parameters") == 0)
		{
			std::string params = line.substr(first + 1, last - first - 1);
			net_data.insert({ "parameters", params });
		}

	}

	// Parse layers
	std::vector<int> layers;
	if (net_data.find("layers") != net_data.end())
	{
		int layer_size;
		std::stringstream iss(net_data["layers"]);
		while (iss >> layer_size)
			layers.push_back(layer_size);
	}
	else
	{
		throw "layers key not found while parsing net data";
	}

	// Parse activation
	Activation hidden_activation;
	Activation output_activation;
	if (net_data.find("activations") != net_data.end())
	{
		std::string hidden_activation_str;
		std::string output_activation_str;
		std::stringstream iss(net_data["activations"]);
		iss >> hidden_activation_str >> output_activation_str;

		hidden_activation = str_to_activation(hidden_activation_str);
		output_activation = str_to_activation(output_activation_str);
	}
	else
	{
		throw "activations key not found while parsing net data";
	}

	// Parse parameters
	std::vector<float> parameters;
	if (net_data.find("parameters") != net_data.end())
	{
		float param;
		std::stringstream iss(net_data["parameters"]);
		while (iss >> param)
			parameters.push_back(param);
	}
	else
	{
		throw "parameters key not found while parsing net data";
	}

	// Build net
	build(layers, hidden_activation, output_activation);

	// Set parameters
	set_parameters(parameters);
}