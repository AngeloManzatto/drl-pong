#pragma once

#include <iostream>

namespace ann {

	/**
	 * Prints a multidimensional tensor on console
	 *
	 * @param[in] rank - Number of dimensions. Ex: 1,2,3,4.
	 * @param[in] shapes - Number of elements inside each dimension. Ex: {2,3,4}.
	 * @param[in] data - Elements in vectorized format. Ex: {1,2,3,4,5,6,7...}.
	 * @param[in] offset and dimension_step are just for recursion purposes and should not be used.
	*/
	inline void print_tensor(const int rank, const int * shapes, const float * data, int offset = 0, int dimension_step = 0)
	{
		int i;
		if (rank == 0)
			return;

		// Recursive step until finds tensor last dimension
		if (dimension_step < rank - 1)
		{
			for (i = 0; i < shapes[dimension_step]; i++)
			{
				print_tensor(rank, shapes, data, (i + offset) * shapes[dimension_step + 1], dimension_step + 1);
			}
			std::cout << "\n";
		}
		else
		{
			for (i = 0; i < shapes[dimension_step]; i++)
			{
				if (i != 0) // Avoid inserting comma after printing the last element.
				{
					std::cout << ", " << *(data + offset + i);
				}
				else
				{
					std::cout << *(data + offset + i);
				}
			}
			std::cout << "\n";
		}
	}

	/**
	 * Converts an image with dimensions (C,H,W) into columns with dimension (C * K_H * K_W, O_H * O_W)

	   This operation is mainly used for convolution layers to threat them as a simple matrix multiplication.

	   @Warning: The out_h and out_w paramters should be computed using the formula below!

	 * @param[in] image - Image in vectorized format.
	 * @param[in] c, in_h, in_w - channels, hight and width of input image
	 * @param[in] out_h = int((w + 2 * p_w - k_w - (k_w - 1) * (d_w - 1)) / s_w) + 1
	 * @param[in] out_w = int((h + 2 * p_h - k_h - (k_h - 1) * (d_h - 1)) / s_h) + 1
	 * @param[in] k_h, k_w - Kernel size
	 * @param[in] s_h, s_w - Stride size
	 * @param[in] p_h, p_w - Padding size
	 * @param[in] d_h, d_w - Dilation size
	 * @param[out] column - Image converted in column format [i_c * k_h * k_w, o_h * o_w]
	*/
	inline void im2col(
		const float * image,
		int c,
		int in_h, int in_w,
		int out_h, int out_w,
		int k_h, int k_w,
		int s_h, int s_w,
		int p_h, int p_w,
		int d_h, int d_w,
		float * column
	)
	{
		int idx_c, idx_w, idx_h, f_w, f_h, index_w, index_h, col_i, col_j, col_idx, img_idx, image_area, column_area;

		image_area = in_h * in_w;
		column_area = k_w * k_h * out_w * out_h;

		for (idx_c = 0; idx_c < c; idx_c++)
		{

			col_idx = 0;
			col_i = 0;
			for (idx_h = -p_h, index_h = 0; index_h < out_h; idx_h += s_h, index_h++)
			{
				for (idx_w = -p_w, index_w = 0; index_w < out_w; idx_w += s_w, index_w++)
				{
					col_j = 0;
					for (f_h = 0; f_h < k_h; f_h++)
					{

						for (f_w = 0; f_w < k_w; f_w++)
						{

							col_idx = col_j * out_h * out_w + col_i + column_area * idx_c;
							img_idx = (idx_h + d_h * f_h) * in_w + (idx_w + d_w * f_w) + image_area * idx_c;

							if (idx_w + d_w * f_w < 0 || idx_w + d_w * f_w >= in_w || idx_h + d_h * f_h < 0 || idx_h + d_h * f_h >= in_h)
							{
								column[col_idx] = 0;

							}
							else
							{
								column[col_idx] = image[img_idx];
							}
							col_idx++;
							col_j++;
						}

					}
					col_i++;
				}

			}
		}
	}

	/**
	 * Converts a column with dimensions (C * K_H * K_W, O_H * O_W) back as an image with dimensions (C,H,W)

	   This operation is mainly used for convolution layers to threat them as a simple matrix multiplication.

	   @Warning: The out_h and out_w paramters should be computed using the formula below!

	 * @param[in] column - Image converted in column format
	 * @param[in] c, in_h, in_w - channels, hight and width of input image
	 * @param[in] out_h = int((w + 2 * p_w - k_w - (k_w - 1) * (d_w - 1)) / s_w) + 1
	 * @param[in] out_w = int((h + 2 * p_h - k_h - (k_h - 1) * (d_h - 1)) / s_h) + 1
	 * @param[in] k_h, k_w - Kernel size
	 * @param[in] s_h, s_w - Stride size
	 * @param[in] p_h, p_w - Padding size
	 * @param[in] d_h, d_w - Dilation size
	 * @param[out] image - Image in vectorized format.
	*/
	inline void col2im(
		const float * column,
		int c,
		int in_h, int in_w,
		int out_h, int out_w,
		int k_h, int k_w,
		int s_h, int s_w,
		int p_h, int p_w,
		int d_h, int d_w,
		float * image
	)
	{
		int idx_c, idx_w, idx_h, f_w, f_h, index_w, index_h, col_i, col_j, col_idx, img_idx, image_area, column_area;

		image_area = in_h * in_w;
		column_area = k_w * k_h * out_w * out_h;

		// Reset values for image
		memset(image, 0, c * in_h * in_w * sizeof(float));

		for (idx_c = 0; idx_c < c; idx_c++)
		{
			col_idx = 0;
			col_i = 0;
			for (idx_h = -p_h, index_h = 0; index_h < out_h; idx_h += s_h, index_h++)
			{
				for (idx_w = -p_w, index_w = 0; index_w < out_w; idx_w += s_w, index_w++)
				{
					col_j = 0;
					for (f_h = 0; f_h < k_h; f_h++)
					{

						for (f_w = 0; f_w < k_w; f_w++)
						{

							col_idx = col_j * out_h * out_w + col_i + column_area * idx_c;
							img_idx = (idx_h + d_h * f_h) * in_w + (idx_w + d_w * f_w) + image_area * idx_c;

							if (idx_w + d_w * f_w >= 0 && idx_w + d_w * f_w < in_w && idx_h + d_h * f_h >= 0 && idx_h + d_h * f_h < in_h)
							{
								image[img_idx] += column[col_idx];

							}

							col_idx++;
							col_j++;
						}

					}
					col_i++;
				}
			}
		}
	}

	/**
	 * y = alpha * x + beta * y
	 *
	 * @param[in] n - Number of elements in input tensor.
	 * @param[in] alpha - Scalar alpha.
	 * @param[in] x - Input tensor.
	 * @param[in] inc_x - Storage spacing between elements of x.
	 * @param[in] beta - Scalar beta. Use 0.0 to set y or 1.0 to accumulate.
	 * @param[out] y - Output tensor.
	 * @param[in] inc_y - Storage spacing between elements of y.
	*/
	inline void axpby(
		const int n,
		const float alpha, const float *x, const int inc_x,
		const float beta, float *y, const int inc_y)
	{
		for (int i = 0; i < n; i++, x += inc_x, y += inc_y)
		{
			*y = alpha * *x + beta * *y;
		}
	}

	/**
	 * General Matrix Multiplication
	 *
	 * C = alpha * A * B + beta * C
	 *
	 * @param[in] transA - Specifies if matrix A is normal ('n') or transposed (any other char).
	 * @param[in] transB - Specifies if matrix B is normal ('n') or transposed (any other char).
	 * @param[in] m - Specifies the number of rows of matrix A and of the matrix C.
	 * @param[in] n - Specifies the number of columns of matrix B and of the matrix C.
	 * @param[in] k - Specifies the number of columns of matrix A and rows of the matrix B.
	 * @param[in] alpha - Scalar alpha.
	 * @param[in] A - Input matrix A.
	 * @param[in] lda - Specifies the first dimension of A. When transA = 'n' then
		   lda must be at least max( 1, m ), otherwise lda must be at least  max( 1, k ).
	 * @param[in] B - Input matrix B.
	 * @param[in] ldb - Specifies the first dimension of B. When transB = 'n' then
		   ldb must be at least max( 1, k ), otherwise ldb must be at least  max( 1, n ).
	 * @param[in] beta - Scalar beta. Use 0.0 to set y or 1.0 to accumulate.
	 * @param[out] C - Output matrix C.
	 * @param[in] ldc - Specifies the first dimension of C. Ldc must be at least max( 1, m ).
	*/
	inline void gemm(
		const char transA, const char transB,
		const int m, const int n, const int k,
		const float alpha,
		const float *A, const int lda,
		const float *B, const int ldb,
		const float beta,
		float * C, const int ldc)
	{
		int i, j, l, ncola, nrowa, nrowb;
		float sum;

		if (transA == 'n')
		{
			nrowa = m;
			ncola = k;
		}
		else
		{
			nrowa = k;
			ncola = m;
		}

		if (transB == 'n')
		{
			nrowb = k;
		}
		else
		{
			nrowb = n;
		}

		if (alpha == 0)
		{
			if (beta == 0)
			{
				for (j = 0; j < n; j++)
				{
					for (i = 0; i < m; i++)
					{
						C[i * ldc + j] = 0;
					}
				}
			}
			else
			{
				for (j = 0; j < n; j++)
				{
					for (i = 0; i < m; i++)
					{
						C[i * ldc + j] = beta * C[i * ldc + j];
					}
				}
			}
		}

		if (transB == 'n')
		{
			if (transA == 'n')
			{
				for (j = 0; j < n; j++)
				{
					if (beta == 0)
					{
						for (i = 0; i < m; i++)
						{
							C[i * ldc + j] = 0;
						}
					}
					else if (beta != 1)
					{
						for (i = 0; i < m; i++)
						{
							C[i * ldc + j] = beta * C[i * ldc + j];
						}
					}

					for (l = 0; l < k; l++)
					{
						sum = alpha * B[l * ldb + j];
						for (i = 0; i < m; i++)
						{
							C[i * ldc + j] = C[i * ldc + j] + sum * A[i*lda + l];
						}
					}
				}
			}
			else
			{
				for (j = 0; j < n; j++)
				{
					for (i = 0; i < m; i++)
					{
						sum = 0;
						for (l = 0; l < k; l++)
						{
							sum = sum + A[l * lda + i] * B[l * ldb + j];
						}
						if (beta == 0)
						{
							C[i * ldc + j] = alpha * sum;
						}
						else
						{
							C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
						}
					}
				}
			}
		}
		else
		{
			if (transA == 'n')
			{
				for (j = 0; j < n; j++)
				{
					if (beta == 0)
					{
						for (i = 0; i < m; i++)
						{
							C[i * ldc + j] = 0;
						}
					}
					else if (beta != 1)
					{
						for (i = 0; i < m; i++)
						{
							C[i * ldc + j] = beta * C[i * ldc + j];
						}
					}

					for (l = 0; l < k; l++)
					{
						sum = alpha * B[j * ldb + l];
						for (i = 0; i < m; i++)
						{
							C[i * ldc + j] = C[i * ldc + j] + sum * A[i * lda + l];
						}
					}
				}
			}
			else
			{
				for (j = 0; j < n; j++)
				{
					for (i = 0; i < m; i++)
					{
						sum = 0;
						for (l = 0; l < k; l++)
						{
							sum = sum + A[l * lda + i] * B[j * ldb + l];
						}
						if (beta == 0)
						{
							C[i * ldc + j] = alpha * sum;
						}
						else
						{
							C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
						}
					}
				}
			}
		}
	}

	/**
	 * Mean over a multidimensional tensor
	 *
	 * mu = sum(x) / scale
	 *
	 * @param[in] x - Input tensor
	 * @param[in] n - Number of elements in input tensor.
	 * @param[in] shape - Number of elements in selected dimension.
	 * @param[in] stride - Number of elements to "skip" to get into the next element on this dimension.
	 * @param[out] mu - Output tensor containing mean over selected shape.
	*/
	inline void mean(
		const float * x,
		const int n,
		const int shape,
		const int stride,
		float * mu
	)
	{
		int i, j, k, slice, index_i;

		slice = int(n / (shape * stride));

		float scale = 1.0f / (slice * stride);

		for (i = 0; i < shape; i++)
		{
			mu[i] = 0;
			for (j = 0; j < slice; j++)
			{
				for (k = 0; k < stride; k++)
				{
					index_i = j * stride * shape + i * stride + k;

					mu[i] += x[index_i];
				}
			}
			mu[i] *= scale;
		}
	}

	/**
	 * Variance over a multidimensional tensor
	 *
	 * mu = var(x) / scale
	 *
	 * @param[in] x - Input tensor
	 * @param[in] mu - Input tensor containing the mean of x.
	 * @param[in] n - Number of elements in input tensor.
	 * @param[in] shape - Number of elements in selected dimension.
	 * @param[in] stride - Number of elements to "skip" to get into the next element on this dimension.
	 * @param[out] var - Output tensor containing variance over selected shape.
	*/
	inline void variance(
		const float * x,
		const float * mu,
		const int n,
		const int shape,
		const int stride,
		float * var)
	{
		int i, j, k, slice, index_i;

		slice = int(n / (shape * stride));

		float scale = 1.0f / (slice * stride);

		for (i = 0; i < shape; i++)
		{
			var[i] = 0;
			for (j = 0; j < slice; j++)
			{
				for (k = 0; k < stride; k++)
				{
					index_i = j * stride * shape + i * stride + k;

					var[i] += pow((x[index_i] - mu[i]), 2);
				}

			}
			var[i] *= scale;
		}
	}

	/**
	 * Normalization over a multidimensional tensor
	 *
	 * norm = (x - mean) / sqr(var)
	 *
	 * @param[in] x - Input tensor
	 * @param[in] mu - Input tensor containing the mean of x.
	 * @param[in] var - Input tensor containing the variance of x.
	 * @param[in] n - Number of elements in input tensor.
	 * @param[in] shape - Number of elements in selected dimension.
	 * @param[in] stride - Number of elements to "skip" to get into the next element on this dimension.
	 * @param[in] eps - Value added to denomination to maintain numerical stability
	 * @param[out] norm - Output tensor containing the normalized values over selected shape.
	*/
	inline void normalize(
		const float * x,
		const float * mu,
		const float * var,
		const int n,
		const int shape,
		const int stride,
		const float eps,
		float * norm)
	{
		int i, j, k, slice, index_i;

		slice = int(n / (shape * stride));

		for (i = 0; i < slice; i++)
		{
			for (j = 0; j < shape; j++)
			{
				for (k = 0; k < stride; k++)
				{
					index_i = i * stride * shape + j * stride + k;

					norm[index_i] = (x[index_i] - mu[j]) / sqrt(var[j] + eps);
				}
			}
		}
	}

	/**
	 * Center data over a multidimensional tensor using gamma and beta
	 *
	 * y = norm * gamma + beta
	 *
	 * @param[in] norm - Normalized input tensor
	 * @param[in] gamma - Input tensor to scale the normalized tensor.
	 * @param[in] beta - Input tensor to offset the normalized tensor.
	 * @param[in] n - Number of elements in input tensor.
	 * @param[in] shape - Number of elements in selected dimension.
	 * @param[in] stride - Number of elements to "skip" to get into the next element on this dimension.
	 * @param[out] y - Output tensor containing the centered data over selected shape.
	*/
	inline void center(
		const float * norm,
		const float * gamma,
		const float * beta,
		const int n,
		const int shape,
		const int stride,
		float * y)
	{
		int i, j, k, slice, index_i;

		slice = int(n / (shape * stride));

		// y = gamma * x + beta
		for (i = 0; i < slice; i++)
		{
			for (j = 0; j < shape; j++)
			{
				for (k = 0; k < stride; k++)
				{
					index_i = i * stride * shape + j * stride + k;

					y[index_i] = gamma[j] * norm[index_i] + beta[j];
				}
			}
		}
	}

	/**
	 * Concatenate layer on feed forward operation
	 * y = [x1, x2,...xn]
	 *
	 * @param[in] n_tensors - Number of tensors inside vector x.
	 * @param[in] n - Number of elements by tensor.
	 * @param[in] x - Input vector of tensors of size [n_tensors , n]
	 * @param[out] y - Output tensor of size [n_tensors * n]
	*/
	inline void concat_layer_forward(const int n_tensors, const int n, const float ** x, float * y)
	{
		for (int i = 0; i < n_tensors; i++)
		{
			memcpy(y + i * n, x[i], n * sizeof(float));
		}
	}

	/**
	 * Concatenate layer on backpropagation operation
	 * y = [x1, x2,...xn]
	 *
	 * @param[in] n_tensors - Number of tensors inside vector x.
	 * @param[in] n - Number of elements by tensor.
	 * @param[in] dy - Gradient of output tensor of size [n_tensors * n]
	 * @param[out] dx - Gradient vectors of tensors of size [n_tensors , n]
	*/
	inline void concat_layer_backward(const int n_tensors, const int n, const float * dy, float ** dx)
	{
		for (int i = 0; i < n_tensors; i++)
		{
			memcpy(dx[i], dy + i * n, n * sizeof(float));
		}
	}

	/**
	 * Add layer on feed forward operation
	 * y = x1 + alpha * x2
	 *
	 * @param[in] n - Number of elements by tensor.
	 * @param[in] alpha - Scaler for the second tensor
	 * @param[in] x1 - First input tensor
	 * @param[in] x2 - Second input tensor
	 * @param[out] y - Output tensor
	*/
	inline void add_layer_forward(const int n, const float alpha, const float * x1, const float * x2, float * y)
	{
		for (int i = 0; i < n; i++)
		{
			y[i] = x1[i] + alpha * x2[i];
		}
	}

	/**
	 * Add layer on backpropagation operation
	 * y = x1 + alpha * x2
	 *
	 * @param[in] n - Number of elements by tensor.
	 * @param[in] alpha - Scaler for the second tensor
	 * @param[in] dy - Output gradient tensor
	 * @param[out] dx1 - First gradient input tensor
	 * @param[out] dx2 - Second gradient input tensor
	*/
	inline void add_layer_backward(const int n, const float alpha, const float * dy, float * dx1, float *dx2)
	{
		for (int i = 0; i < n; i++)
		{
			dx1[i] = dy[i];
			dx2[i] = alpha * dy[i];
		}
	}

	/**
	 * Fully connected layer on feed forward operation
	 * y = x * w + b
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] n_inputs - Number of input neurons.
	 * @param[in] n_outputs - Number of output neurons.
	 * @param[in] x - Input tensor of shapes [batch, n_inputs]
	 * @param[in] w - Learnable weights of shapes [n_inputs, n_outputs]
	 * @param[in] b - Learnable bias of shapes [1, n_outputs]
	 * @param[out] y - Output tensor of shapes [batch, n_outputs]
	*/
	inline void fc_layer_forward(
		const int batch, const int n_inputs, const int n_outputs,
		const float * x,
		const float * w,
		const float * b,
		float * y
	)
	{
		int i, m, n, k;

		m = batch;
		n = n_outputs;
		k = n_inputs;

		// y[m,n] = x[m,k] * w[k,n]
		gemm('n', 'n', m, n, k, 1, x, k, w, n, 0, y, n);

		// y[m,n] = y[m,n] + b[1,n]
		if (b)
		{
			for (i = 0; i < m; i++)
			{
				axpby(n, 1, b, 1, 1, y + i * n, 1);
			}
		}

	}

	/**
	 * Fully connected layer on backpropagation operation
	 * dw = x.T * dy
	 * db = sum(dy, axis=1)
	 * dx = dy * w.T
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] n_inputs - Number of input neurons.
	 * @param[in] n_outputs - Number of output neurons.
	 * @param[in] x - Input tensor of shapes [batch, n_inputs]
	 * @param[in] w - Learnable weights of size [n_inputs, n_outputs]
	 * @param[in] dy - Gradient of output tensor of shapes [batch, n_outputs]
	 * @param[out] dw - Gradient of learnable weights of shapes [n_inputs, n_outputs]
	 * @param[out] db - Gradient of learnable bias of shapes[1, n_outputs]
	 * @param[out] dx - Gradient of input tensor of shapes [batch, n_inputs]
	*/
	inline void fc_layer_backward(
		const int batch, const int n_inputs, const int n_outputs,
		const float * x,
		const float * w,
		const float * dy,
		float * dw, float * db, float * dx
	)
	{
		int i, m, n, k;

		m = n_inputs;
		n = n_outputs;
		k = batch;

		if (db)
		{
			for (i = 0; i < k; i++)
			{
				axpby(n, 1, dy + i * n, 1, 1, db, 1);
			}
		}

		// dw[m, n] = x[k, m].T * dy[k, n]
		gemm('t', 'n', m, n, k, 1, x, m, dy, n, 1, dw, n);

		// dx[k, m] = dy[k, n] * w[m, n].T
		gemm('n', 't', k, m, n, 1, dy, n, w, n, 0, dx, m);
	}

	/**
	 * Convolution 2D layer on feed forward operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] in_c, in_h, in_w - Channels, height and width of input image.
	 * @param[in] out_c, out_h, out_w - Channels, height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, in_c, in_h, in_w]
	 * @param[in] w - 4D Tensor of learnable weights of shapes [out_c, in_c, k_h , k_w]
	 * @param[in] b - 1D Tensor of bias of shapes [out_c]
	 * @param[in] cols - 2D Tensor of size [in_c * k_h * k_w, out_h * out_w]
	 * @param[out] y - Output tensor of shapes [batch, out_c, out_h, out_w]

	 * out_h = int((in_h + 2 * p_h - k_h - (k_h - 1) * (d_h - 1)) / s_h) + 1;
	 * out_w = int((in_w + 2 * p_w - k_w - (k_w - 1) * (d_w - 1)) / s_w) + 1;

	*/
	inline void conv2d_layer_forward(
		const int batch,
		const int in_c, const int in_h, const int in_w,
		const int out_c, const int out_h, const int out_w,
		const int k_h, const int k_w,
		const int s_h, const int s_w,
		const int p_h, const int p_w,
		const int d_h, const int d_w,
		const float * x,
		const float * w,
		const float * b,
		float * cols,
		float * y)
	{
		int i, m, n, k, b_i, in_stride, out_stride;

		m = out_c;
		n = out_h * out_w;
		k = in_c * k_h * k_w;
		in_stride = in_c * in_h * in_w;
		out_stride = out_c * out_h * out_w;

		for (b_i = 0; b_i < batch; b_i++)
		{
			// Image to Colum conversion
			im2col(x + b_i * in_stride,
				in_c, in_h, in_w,
				out_h, out_w,
				k_h, k_w,
				s_h, s_w,
				p_h, p_w,
				d_h, d_w,
				cols);

			// Matrix multiplication 
			gemm('n', 'n', m, n, k, 1, w, k, cols, n, 0, y + b_i * out_stride, n);

			// Add bias
			if (b)
			{
				for (i = 0; i < out_c; i++)
				{
					axpby(n, 1, b + i, 0, 1, y + i * n + b_i * out_stride, 1);
				}
			}
		}
	}

	/**
	 * Convolution 2D layer on backpropagation operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] in_c, in_h, in_w - Channels, height and width of input image.
	 * @param[in] out_c, out_h, out_w - Channels, height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, in_c, in_h, in_w]
	 * @param[in] w - 4D Tensor of learnable weights of shapes [out_c, in_c, k_h , k_w]
	 * @param[in] b - 1D Tensor of bias of shapes [out_c]
	 * @param[in] cols - 2D Tensor of size [in_c * k_h * k_w, out_h * out_w]
	 * @param[in] dy - Gradient of output tensor of shapes [batch, out_c, out_h, out_w]
	 * @param[out] dw - Gradient of learnable weights of shapes [out_c, in_c, k_h , k_w]
	 * @param[out] db - Gradient of learnable bias of shapes[out_c]
	 * @param[out] dx - Gradient of input tensor of shapes [batch, in_c, in_h, in_w]

	 * out_h = int((in_h + 2 * p_h - k_h - (k_h - 1) * (d_h - 1)) / s_h) + 1;
	 * out_w = int((in_w + 2 * p_w - k_w - (k_w - 1) * (d_w - 1)) / s_w) + 1;

	*/
	inline void conv2d_layer_backward(
		const int batch,
		const int in_c, const int in_h, const int in_w,
		const int out_c, const int out_h, const int out_w,
		const int k_h, const int k_w,
		const int s_h, const int s_w,
		const int p_h, const int p_w,
		const int d_h, const int d_w,
		const float * x,
		const float * w,
		const float * b,
		float * cols,
		const float * dy, float * dw, float * db, float * dx
	)
	{
		int m, n, k, b_i, o_i, in_stride, out_stride;

		m = out_c;
		n = in_c * k_h * k_w;
		k = out_h * out_w;
		in_stride = in_c * in_h * in_w;
		out_stride = out_c * out_h * out_w;

		if (db)
		{
			for (b_i = 0; b_i < batch; b_i++)
			{
				for (o_i = 0; o_i < out_c; o_i++)
				{
					axpby(k, 1, dy + o_i * k + b_i * out_stride, 1, 1, db + o_i, 0);
				}
			}
		}

		for (b_i = 0; b_i < batch; b_i++)
		{
			// Image to Column conversion
			im2col(x + b_i * in_stride,
				in_c, in_h, in_w,
				out_h, out_w,
				k_h, k_w,
				s_h, s_w,
				p_h, p_w,
				d_h, d_w,
				cols);

			// dw[k,n] = dot(dcosts[m,n], col_.T[n,k])
			gemm('n', 't', m, n, k, 1, dy + b_i * out_stride, k, cols, k, 1, dw, n);

			// dx[m,k] = dot(w[m,n], dy[k,n].T)
			gemm('t', 'n', n, k, m, 1, w, n, dy + b_i * out_stride, k, 0, cols, k);

			// Column to image
			col2im(cols,
				in_c, in_h, in_w,
				out_h, out_w,
				k_h, k_w,
				s_h, s_w,
				p_h, p_w,
				d_h, d_w,
				dx + b_i * in_stride);
		}

	}

	/**
	 * Deconvolution 2D layer on feed forward operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] in_c, in_h, in_w - Channels, height and width of input image.
	 * @param[in] out_c, out_h, out_w - Channels, height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, in_c, in_h, in_w]
	 * @param[in] w - 4D Tensor of learnable weights of shapes [in_c, out_c, k_h, k_w]
	 * @param[in] b - 1D Tensor of bias of shapes [out_c]
	 * @param[in] cols - 2D Tensor of size [out_c * k_h * k_w, in_h * in_w]
	 * @param[out] y - Output tensor of shapes [batch, out_c, out_h, out_w]

	 * out_h = int((in_w - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + 1);
	 * out_w = int((in_h - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + 1);

	*/
	inline void deconv2d_layer_forward(
		const int batch,
		const int in_c, const int in_h, const int in_w,
		const int out_c, const int out_h, const int out_w,
		const int k_h, const int k_w,
		const int s_h, const int s_w,
		const int p_h, const int p_w,
		const int d_h, const int d_w,
		const float * x,
		const float * w,
		const float * b,
		float * cols,
		float * y)
	{
		int i, b_i, m, n, k, in_stride, out_stride;

		m = out_c * k_h * k_w;
		n = in_h * in_w;
		k = in_c;
		in_stride = in_c * in_h * in_w;
		out_stride = out_c * out_h * out_w;

		for (b_i = 0; b_i < batch; b_i++)
		{
			// cols[out_c * k_h * k_w, in_h * in_w] = w[out_c * k_h * k_w, in_c].T * x[in_c , in_h * in_w]
			gemm('t', 'n', m, n, k, 1, w, m, x + b_i * in_stride, n, 0, cols, n);

			// Column to image
			col2im(cols,
				out_c,
				out_h, out_w,
				in_h, in_w,
				k_h, k_w,
				s_h, s_w,
				p_h, p_w,
				d_h, d_w,
				y + b_i * out_stride);

			// Add bias
			if (b)
			{
				for (i = 0; i < out_c; i++)
				{
					axpby(n, 1, b + i, 0, 1, y + i * n + b_i * out_stride, 1);
				}
			}
		}
	}

	/**
	 * Deconvolution 2D layer on backpropagation operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] in_c, in_h, in_w - Channels, height and width of input image.
	 * @param[in] out_c, out_h, out_w - Channels, height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, in_c, in_h, in_w]
	 * @param[in] w - 4D Tensor of learnable weights of shapes [in_c, out_c, k_h, k_w]
	 * @param[in] b - 1D Tensor of bias of shapes [out_c]
	 * @param[in] cols - 2D Tensor of size [out_c * k_h * k_w * in_h * in_w]
	 * @param[in] dy - Gradient of output tensor of shapes [batch, out_c, out_h, out_w]
	 * @param[out] dw - Gradient of learnable weights of shapes [in_c, out_c, k_h, k_w]
	 * @param[out] db - Gradient of learnable bias of shapes[out_c]
	 * @param[out] dx - Gradient of input tensor of shapes [batch, in_c, in_h, in_w]

	 * out_h = int((in_h + 2 * p_h - k_h - (k_h - 1) * (d_h - 1)) / s_h) + 1;
	 * out_w = int((in_w + 2 * p_w - k_w - (k_w - 1) * (d_w - 1)) / s_w) + 1;

	*/
	inline void deconv2d_layer_backward(
		const int batch,
		const int in_c, const int in_h, const int in_w,
		const int out_c, const int out_h, const int out_w,
		const int k_h, const int k_w,
		const int s_h, const int s_w,
		const int p_h, const int p_w,
		const int d_h, const int d_w,
		const float * x,
		const float * w,
		const float * b,
		float * cols,
		const float * dy, float * dw, float * db, float * dx
	)
	{
		int m, n, k, b_i, o_i, in_stride, out_stride;

		in_stride = in_c * in_h * in_w;
		out_stride = out_c * out_h * out_w;

		k = in_h * in_w;

		if (db)
		{
			for (b_i = 0; b_i < batch; b_i++)
			{
				for (o_i = 0; o_i < out_c; o_i++)
				{
					axpby(k, 1, dy + o_i * k + b_i * out_stride, 1, 1, db + o_i, 0);
				}
			}
		}

		for (b_i = 0; b_i < batch; b_i++)
		{
			m = in_c;
			n = out_c * k_h * k_w;
			k = in_h * in_w;

			// dy[out_c * out_h * out_w] -> cols[out_c * k_h * k_w * in_h * in_w]
			im2col(dy + b_i * out_stride,
				out_c,
				out_h, out_w,
				in_h, in_w,
				k_h, k_w,
				s_h, s_w,
				p_h, p_w,
				d_h, d_w,
				cols);

			// dw[in_c * out_c * k_h * k_w] = x[in_c , in_h * in_w] * cols[out_c * k_h * k_w , in_h * in_w].T
			gemm('n', 't', m, n, k, 1, x + b_i * in_stride, k, cols, k, 1, dw, n);

			m = in_c;
			n = in_h * in_w;
			k = out_c * k_h * k_w;

			// dx[in_c, in_h * in_w] = w[in_c, out_c * k_h * k_w] * cols[out_c * k_h * k_w , in_h * in_w]
			gemm('n', 'n', m, n, k, 1, w, k, cols, n, 1, dx + b_i * in_stride, n);

		}

	}

	/**
	 * Max Pooling 2D layer on feed forward operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] channels - Number of input channels (same as output)
	 * @param[in] in_h, in_w - Height and width of input image.
	 * @param[in] out_h, out_w - Height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, channels, in_h, in_w]
	 * @param[in] indices - 1D Tensor of size [batch * channels * out_h * out_w]
	 * @param[out] y - 4D tensor with shapes [batch, channels, out_h, out_w]

	 * out_h = int(((in_h + 2 * p_h - d_h * (k_h - 1) - 1) / s_h) + 1);
	 * out_w = int(((in_w + 2 * p_w - d_w * (k_w - 1) - 1) / s_w) + 1);

	*/
	inline void maxpool2d_layer_forward(
		const int batch, const int channels,
		const int in_h, const int in_w,
		const int out_h, const int out_w,
		const int k_h, const int k_w,
		const int s_h, const int s_w,
		const int p_h, const int p_w,
		const int d_h, const int d_w,
		const float * x,
		float * indices,
		float * y)
	{
		int b_i, c_i, index_h, index_w, pi_h, pi_w, ki_h, ki_w, idx_input, idx_output;

		float max_value;
		int max_index;

		// Loop over batch
		for (b_i = 0; b_i < batch; b_i++)
		{
			// Loop over channel
			for (c_i = 0; c_i < channels; c_i++)
			{
				// Loop over height
				for (pi_h = -p_h, index_h = 0; index_h < out_h; pi_h += s_h, index_h++)
				{
					// Loop over width
					for (pi_w = -p_w, index_w = 0; index_w < out_w; pi_w += s_w, index_w++)
					{
						// Single Pooling
						max_value = std::numeric_limits<float>::lowest();

						for (ki_h = 0; ki_h < k_h; ki_h++)
						{
							for (ki_w = 0; ki_w < k_w; ki_w++)
							{
								idx_input = (d_h * ki_h + pi_h) * in_w +
									(d_w * ki_w + pi_w) +
									c_i * in_h * in_w +
									b_i * channels * in_h * in_w;

								if (d_h * ki_h + pi_h >= 0 &&
									d_h * ki_h + pi_h < in_h &&
									d_w * ki_w + pi_w >= 0 &&
									d_w * ki_w + pi_w < in_w)
								{
									if (x[idx_input] > max_value)
									{
										max_value = x[idx_input];
										max_index = idx_input;
									}
								}
							}
						}

						idx_output = index_w +
							index_h * out_w +
							c_i * out_h * out_w +
							b_i * channels * out_h * out_w;

						y[idx_output] = max_value;

						indices[idx_output] = static_cast<float>(max_index);
					}
				}
			}
		}
	}

	/**
	 * Max Pooling 2D layer on backpropagation operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] channels - Number of input channels (same as output)
	 * @param[in] in_h, in_w - Height and width of input image.
	 * @param[in] out_h, out_w - Height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, channels, in_h, in_w]
	 * @param[in] indices - 1D Tensor of size [batch * channels * out_h * out_w]
	 * @param[in] dy - 4D tensor with shapes [batch, channels, out_h, out_w]
	 * @param[out] dx - 4D tensor with shapes [batch, channels, in_h, in_w]

	 * out_h = int(((in_h + 2 * p_h - d_h * (k_h - 1) - 1) / s_h) + 1);
	 * out_w = int(((in_w + 2 * p_w - d_w * (k_w - 1) - 1) / s_w) + 1);

	*/
	inline void maxpool2d_layer_backward(
		const int batch, const int channels,
		const int in_h, const int in_w,
		const int out_h, const int out_w,
		const int k_h, const int k_w,
		const int s_h, const int s_w,
		const int p_h, const int p_w,
		const int d_h, const int d_w,
		const float * indices,
		const float * dy,
		float * dx)
	{
		int b_i, c_i, index_h, index_w, pi_h, pi_w, ki_h, ki_w, idx_input, idx_filter;

		// Loop over batch
		for (b_i = 0; b_i < batch; b_i++)
		{
			// Loop over channel
			for (c_i = 0; c_i < channels; c_i++)
			{
				int index = 0;
				// Loop over height
				for (pi_h = -p_h, index_h = 0; index_h < out_h; pi_h += s_h, index_h++)
				{
					// Loop over width
					for (pi_w = -p_w, index_w = 0; index_w < out_w; pi_w += s_w, index_w++)
					{

						for (ki_h = 0; ki_h < k_h; ki_h++)
						{
							for (ki_w = 0; ki_w < k_w; ki_w++)
							{
								idx_input = (d_h * ki_h + pi_h) * in_w +
									(d_w * ki_w + pi_w) +
									c_i * in_h * in_w +
									b_i * channels * in_h * in_w;

								idx_filter = c_i * out_h * out_w +
									b_i * channels * out_h * out_w +
									index; // output area + output volume + index

								if (d_h *ki_h + pi_h >= 0 &&
									d_h *ki_h + pi_h < in_h &&
									d_w *ki_w + pi_w >= 0 &&
									d_w *ki_w + pi_w < in_w)
								{
									dx[idx_input] += (indices[idx_filter] == idx_input) * dy[idx_filter];
								}

							}
						}
						index++;
					}
				}
			}
		}
	}

	/**
	 * Drop Out layer on feed forward operation
	 *
	 * @param[in] n - Number of elements.
	 * @param[in] prob - Probability threshold of setting neuron to zero (below this value).
	 * @param[in] x - Input tensor
	 * @param[out] y - Output tensor
	 * @param[out] mask - Probability mask for backpropagation
	*/
	inline void dropout_layer_forward(const int n, const float prob, const float * x, float * y, float * mask, bool is_training)
	{
		if (is_training)
		{
			float scale = 1.0f / (1.0f - prob);

			for (int i = 0; i < n; i++)
			{
				float p = (float)rand() / RAND_MAX;
				mask[i] = p;

				if (p < prob) {
					y[i] = 0.0;
				}
				else
				{
					y[i] = x[i] * scale;
				}
			}
		}
		else
		{
			memcpy(y, x, sizeof(float)*n);
		}

	}

	/**
	 * Drop Out layer on backpropagation operation
	 *
	 * @param[in] n - Number of elements.
	 * @param[in] prob - Probability threshold of setting neuron to zero (below this value).
	 * @param[in] dy - Gradient output
	 * @param[in] mask - Probability mask for backpropagation
	 * @param[out] dx - Gradient input
	*/
	inline void dropout_layer_backward(const int n, const float prob, const float * dy, const float * mask, float  * dx)
	{
		float scale = 1.0f / (1.0f - prob);

		for (int i = 0; i < n; i++)
		{
			float p = mask[i];
			if (p < prob) {
				dx[i] = 0.0f;
			}
			else
			{
				dx[i] = dy[i] * scale;
			}
		}
	}

	inline void batch_norm_forward(
		const int n,
		const int shape,
		const int stride,
		const float momentum,
		const float epsilon,
		const float * x,
		float * mu,
		float * var,
		float * norm,
		float * gamma,
		float * beta,
		float * moving_mean,
		float * moving_variance,
		float * y,
		bool is_training)
	{
		if (is_training)
		{
			// Calculat mean
			mean(x, n, shape, stride, mu);

			// Calculate variance
			variance(x, mu, n, shape, stride, var);

			// Calculate moving_mean = (1 - momentum) * mean + momentum * moving_mean  
			axpby(shape, 1.0f - momentum, mu, 1, momentum, moving_mean, 1);

			// Calculate moving_var = (1 - momentum) * var + momentum * moving_var 
			axpby(shape, 1.0f - momentum, var, 1, momentum, moving_variance, 1);

			// Normalize data
			normalize(x, mu, var, n, shape, stride, epsilon, norm);

			// Apply batch normalization
			center(norm, gamma, beta, n, shape, stride, y);
		}
		else
		{
			// Normalize data
			normalize(x, moving_mean, moving_variance, n, shape, stride, epsilon, norm);

			// Apply batch normalization
			center(norm, gamma, beta, n, shape, stride, y);
		}

	}

	inline void batch_norm_backward(
		const int n,
		const int shape,
		const int stride,
		const float * x,
		const float * mu,
		const float * var,
		const float * norm,
		const float * gamma,
		const float * beta,
		float * dmu,
		float * dvar,
		float * dgamma,
		float * dbeta,
		float * dy,
		float * dx
	)
	{
		int i, j, k, index_i;

		float std_inv = 0.0f;
		float x_mean = 0.0f;
		float dx_norm = 0.0f;
		float dx_mean = 0.0f;
		float dmean = 0.0f;
		float dvariance = 0.0f;
		float dg = 0.0f;
		float db = 0.0f;

		int N = n / shape;

		int slice = int(n / (shape * stride));

		// Calculate dgamma and dbeta
		for (i = 0; i < shape; i++)
		{
			dvariance = 0;
			dmean = 0;
			dg = 0;
			db = 0;
			dx_mean = 0;
			std_inv = 1.0f / sqrt(var[i] + 1e-05f);

			for (j = 0; j < slice; j++)
			{
				for (k = 0; k < stride; k++)
				{
					index_i = j * stride * shape + i * stride + k;

					dg += dy[index_i] * norm[index_i];
					db += dy[index_i];

					x_mean = x[index_i] - mu[i];
					dx_norm = dy[index_i] * gamma[i];

					dvariance += dx_norm * x_mean * -0.5f * std_inv * std_inv * std_inv;
					dmean += dx_norm * -std_inv;
					dx_mean += -2.0f * x_mean / slice;
				}

			}

			dgamma[i] = dg;
			dbeta[i] = db;

			dvar[i] = dvariance;
			dmu[i] = dmean + dvariance * dx_mean;

		}

		// Calculate dx
		for (i = 0; i < shape; i++)
		{
			// std_inv = 1 / sqrt(var +eps)
			std_inv = 1.0f / sqrt(var[i] + 1e-5f);

			for (j = 0; j < slice; j++)
			{
				for (k = 0; k < stride; k++)
				{
					index_i = j * stride * shape + i * stride + k;

					// dx_norm = dout * gamma
					dx_norm = dy[index_i] * gamma[i];

					// x_mean = x - mean
					x_mean = x[index_i] - mu[i];

					// dx = dx_norm * std_inv + dvar * 2 * x_mean / batch + dmean / batch
					dx[index_i] += dx_norm * std_inv + (dvar[i] * 2.0f * x_mean) / N + dmu[i] / N;
				}
			}
		}
	}

	/**
	 * Sigmoid activation layer on feed forward operation
	 * y = 1/(1+exp(-x))
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] x - Input tensor in vectorized format
	 * @param[out] y - Output tensor in vectorized format
	*/
	inline void sigmoid_forward(const int n, const float * x, float * y)
	{
		for (int i = 0; i < n; i++, x++, y++)
		{
			*y = 1.0f / (1.0f + expf(-*x));
		}
	}

	/**
	 * Sigmoid activation layer on backpropagation operation
	 * dx = 1/(1+exp(-x)) * (1 - 1/(1+exp(-x))) * dy
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] x - Input tensor in vectorized format
	 * @param[in] dy - Gradient of output tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format
	*/
	inline void sigmoid_backward(const int n, const float * x, const float * dy, float * dx)
	{
		for (int i = 0; i < n; i++, x++, dy++, dx++)
		{
			*dx = 1.0f / (1.0f + expf(-*x)) * (1.0f - 1.0f / (1.0f + expf(-*x))) * * dy;
		}
	}

	/**
	 * Fast sigmoid activation layer on backpropagation operation
	 * dx = y * (1 - y) * dy
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] y - Output tensor of sigmoid forward in vectorized format
	 * @param[in] dy - Gradient of output tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor of shapes [batch, n_inputs]
	*/
	inline void fast_sigmoid_backward(const int n, const float * y, const float * dy, float * dx)
	{
		for (int i = 0; i < n; i++, y++, dy++, dx++)
		{
			*dx = *y * (1.0f - *y) * * dy;
		}
	}

	/**
	 * Tanh activation layer on feed forward operation
	 * y = (e(x) - e(-x))/(e(x) + e(-x))
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] x - Input tensor in vectorized format
	 * @param[out] y - Output tensor in vectorized format
	*/
	inline void tanh_forward(const int n, const float *x, float * y)
	{
		for (int i = 0; i < n; i++, x++, y++)
		{
			*y = tanhf(*x);
		}
	}

	/**
	 * Tanh activation layer on backpropagation operation
	 * dx = (1 - tanh(x)^2) * dy
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] x - Input tensor in vectorized format
	 * @param[in] dy - Gradient of output tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format
	*/
	inline void tanh_backward(const int n, const float *x, const float *dy, float * dx)
	{
		for (int i = 0; i < n; i++, x++, dy++, dx++)
		{
			*dx = (1.0f - tanhf(*x) * tanhf(*x)) * *dy;
		}
	}

	/**
	 * Fast tanh activation layer on backpropagation operation
	 * dx = (1 - y^2) * dy
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] y - Output tensor of tanh forward in vectorized format
	 * @param[in] dy - Gradient of output tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format
	*/
	inline void fast_tanh_backward(const int n, const float *y, const float *dy, float * dx)
	{
		for (int i = 0; i < n; i++, y++, dy++, dx++)
		{
			*dx = (1.0f - *y * *y) * *dy;
		}
	}

	/**
	 * ReLU activation layer on feed forward operation
	 * y = max(0, x)
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] x - Input tensor in vectorized format
	 * @param[out] y - Output tensor in vectorized format
	*/
	inline void relu_forward(const int n, const float *x, float * y)
	{
		for (int i = 0; i < n; i++, x++, y++)
		{
			*y = *x > 0.0f ? *x : 0.0f;
		}
	}

	/**
	 * ReLU activation layer on backpropagation operation
	 * dx = { x > 0 -> 1, x <= 0 -> 0} * dy
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] x - Input tensor in vectorized format
	 * @param[in] dy - Gradient of output tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format
	*/
	inline void relu_backward(const int n, const float * x, const float * dy, float * dx)
	{
		for (int i = 0; i < n; i++, x++, dy++, dx++)
		{
			*dx = (*x > 0.0f ? 1.0f : 0.0f) * *dy;
		}
	}

	/**
	 * Softmax activation layer on feed forward operation
	 * y = e(x)/ sum(e(x))
	 *
	 * @param[in] x - Input tensor in vectorized format
	 * @param[in] rank - Number of dimensions. Ex: 1,2,3,4.
	 * @param[in] shapes - Number of elements inside each dimension. Ex: {2,3,4}.
	 * @param[in] axis - Axis along which the softmax operation will be performed.
	 * @param[out] y - Output tensor in vectorized format
	*/
	inline void softmax_forward(const float * x, const int n, const int shape, const int stride, float * y)
	{
		int i, j, k, ind_x, slice;

		float max, sum;

		slice = int(n / (stride * shape));

		for (i = 0; i < slice; i++)
		{
			for (j = 0; j < stride; j++)
			{
				max = -std::numeric_limits<float>::max();

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					max = (max > x[ind_x]) ? max : x[ind_x];
				}

				sum = 0.0f;

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					sum += exp(x[ind_x] - max);
				}

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					y[ind_x] = exp(x[ind_x] - max) / sum;
				}
			}
		}
	}

	/**
	 * Softmax activation layer on backpropagation operation
	 * dx = dy * jacobian matrix
	 *
	 * @param[in] y - Output tensor from softmax feed forward operation in vectorized format
	 * @param[in] rank - Number of dimensions. Ex: 1,2,3,4.
	 * @param[in] shapes - Number of elements inside each dimension. Ex: {2,3,4}.
	 * @param[in] axis - Axis along which the softmax operation will be performed.
	 * @param[in] dy - Gradient of output tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format
	*/
	inline void softmax_backward(const float * y, const int n, const int shape, const int stride, const float * dy, float * dx)
	{
		int i, j, k, l, ind_dx, ind_y, slice;

		float sum;

		slice = int(n / (stride * shape));

		for (i = 0; i < slice; i++)
		{
			for (j = 0; j < stride; j++)
			{
				// Jacobian matrix
				for (k = 0; k < shape; k++)
				{
					sum = 0.0f;

					ind_dx = i * stride * shape + j + k * stride;

					for (l = 0; l < shape; l++)
					{
						ind_y = i * stride * shape + j + l * stride;

						sum += y[ind_dx] * (1.0f * (k == l) - y[ind_y]) * dy[ind_y];
					}

					dx[ind_dx] = sum;
				}
			}
		}

	}

	/**
	 * Mean Squared Error loss layer on feed forward operation
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] input - Input tensor in vectorized format
	 * @param[in] target - Target tensor in vectorized format

	 * return loss = (input - target)^2
	*/
	inline float mse_loss_layer_forward(const int n, const float * input, const float * target)
	{
		float loss = 0.0f;

		for (int i = 0; i < n; i++)
		{
			loss += (input[i] - target[i]) * (input[i] - target[i]);
		}

		return loss / n;
	}

	/**
	 * Mean Squared Error loss layer on backpropagation operation
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] input - Input tensor in vectorized format
	 * @param[in] target - Target tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format

	 * dx = 2 *(input - target) / n
	*/
	inline void mse_loss_layer_backward(const int n, const float * input, const float * target, float * dx)
	{
		for (int i = 0; i < n; i++)
		{
			dx[i] = (input[i] - target[i]) / 2.0f;
		}
	}

	/**
	 * Cross Entropy loss layer on feed forward operation
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] input - Input tensor in vectorized format
	 * @param[in] target - Target tensor in vectorized format

	 * return loss = -1 * (target * log(input) + (1-target) * log(1-input))
	 *
	 * @Warning: Input values must be between 0 and 1
	*/
	inline float cross_entropy_loss_layer_forward(const int n, const float * input, const float * target)
	{
		float loss = 0.0f;

		for (int i = 0; i < n; i++)
		{
			loss += -1 * (target[i] * log(input[i] + std::numeric_limits<float>::epsilon()) +
				(1 - target[i]) * log(1 - input[i] + std::numeric_limits<float>::epsilon()));
		}

		return loss / n;
	}

	/**
	 * Cross Entropy loss layer on backpropagation operation
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] input - Input tensor in vectorized format
	 * @param[in] target - Target tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format
	 *
	 * dx = -1 * (target / input) + ((1-target)/(1-input))
	*/
	inline void cross_entropy_loss_layer_backward(const int n, const float * input, const float * target, float * dx)
	{
		for (int i = 0; i < n; i++)
		{
			dx[i] = -1 * (target[i] / (input[i] + std::numeric_limits<float>::epsilon())) +
				(1 - target[i]) / (1 - input[i] + std::numeric_limits<float>::epsilon());

			dx[i] /= n;
		}
	}

	/**
	 * Stochastic Gradient Descent Optimizer
	 * w = w - learning rate * g
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] lr - Learning rate
	 * @param[in] mom - Momentum vector
	 * @param[in] g - Gradient vector
	 * @param[in] v - Velocity vector
	 * @param[out] w - Weight vector
	 *
	*/
	inline void sgd_optimizer(
		const int n, const float lr, const float mom,
		const float * g, float * v, float * w
	)
	{
		for (int i = 0; i < n; i++, g++, v++, w++)
		{
			if (mom > 0.0f)
			{
				*v = mom * *v + *g;
				*w = *w - lr * *v;
			}
			else
			{
				*w = *w - lr * *g;
			}

		}
	}

	/**
	 * Root Mean Squared Optimizer
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] lr - Learning rate
	 * @param[in] rho - Discounting factor for the history gradient
	 * @param[in] mom - Momentum vector
	 * @param[in] eps - Constant for numerical stability.
	 * @param[in] g - Gradient vector
	 * @param[in] v - Velocity vector
	 * @param[out] w - Weight vector
	 *
	*/
	inline void rms_prop_optimizer(
		const int n, const float lr, const float rho, const float mom, const float eps,
		const float * g, float * v, float * w
	)
	{
		for (int i = 0; i < n; i++, g++, v++, w++)
		{
			*v = rho * *v + (1.0f - rho) * *g * *g;

			*w = *w - lr * *g / (sqrtf(*v) + eps);
		}
	}

	/**
	 * Adaptive Moment Estimation Optimizer
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] iter - Iteration number.
	 * @param[in] lr - Learning rate
	 * @param[in] beta_1 - Exponential decay rate for the 1st moment estimates.
	 * @param[in] beta_2 - Exponential decay rate for the 2nd moment estimates.
	 * @param[in] eps - Constant for numerical stability.
	 * @param[in] mom - Momentum vector
	 * @param[in] g - Gradient vector
	 * @param[in] v - Velocity vector
	 * @param[in] m - Momentum vector
	 * @param[out] w - Weight vector
	 *
	*/
	inline void adam_optimizer(
		const int n, const int iter, const float lr, const float beta_1, const float beta_2, const float eps,
		const float * g, float * v, float * m, float * w
	)
	{
		float m_corr, v_corr;

		for (int i = 0; i < n; i++, g++, v++, m++, w++)
		{

			*m = beta_1 * *m + (1.0f - beta_1) * *g;

			*v = beta_2 * *v + (1.0f - beta_2) * *g * *g;

			m_corr = *m / (1.0f - powf(beta_1, float(iter)));

			v_corr = *v / (1.0f - powf(beta_2, float(iter)));

			*w = *w - m_corr * lr / (sqrtf(v_corr) + eps);
		}
	}
}