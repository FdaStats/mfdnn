# Multiclass Functional Deep Neural Network Classifier
------------------------------------------------

# Functional data pre-processing
- Given functional data ![first equation](https://latex.codecogs.com/gif.latex?X%28s_1%2C%20%5Cldots%2C%20s_d%29), first use Fourier basis functions to extract projection scores ![second equation](https://latex.codecogs.com/gif.latex?%5Cxi_1%2C%20%5Cxi_2%2C%20%5Cldots) by integration.
-------------------------------------------------------

# Model input and output
- Input: Projection scores ![xi](https://latex.codecogs.com/gif.latex?%5Cxi_1%2C%20%5Cxi_2%2C%20%5Cldots%2C%20%5Cxi_J).
- Output: Binary class label k={1, 2, ..., K}.
-------------------------------------------------------------

# Model selection
## Neural network hyperparameters 
- J: number of projection scores for network inputs
- L: number of layers 
- p: neurons per layer (uniform for all layers)
- s: dropout rate
-------------------------------------------------------------

# Other hyperparameters
- Loss function: softmax loss
- Batch size: data dependent
- Epoch number: data dependent
- Activation function: ReLU
- Optimizer: Adam 
-------------------------------------------------------------

# Function descriptions
## One dimensional functional data
- "mfdnn_1d_par.R": hyperparameter selection with training data. More details can be found in comments
- "mfdnn_1d.R": functional deep neural netowrks. More details can be found in comments 
## Two dimensional functional data
- "mfdnn_2d_par.R": hyperparameter selection with training data. More details can be found in comments
- "mfdnn_2d.R": functional deep neural netowrks. More details can be found in comments 
## Three dimensional functional data
- "mfdnn_3d_par.R": hyperparameter selection with training data. More details can be found in comments
- "mfdnn_3d.R": functional deep neural netowrks. More details can be found in comments 
-------------------------------------------------------------

# Examples
- "example_1d.R": simulated data for one-dimensional functional data
- "example_2d.R": simulated data for two-dimensional functional data
