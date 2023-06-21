
Overview
--------



Part I - Models: Training and Architecture
------------------------------------------

### Day 1: Basics of Neural Nets

#### Plan

- Talk: Course Intro
- Share: Introduce Energy, Fruit, and TicTacToe Problems
- Presentation: Inference script and visualization
- Code: hello world / jupyter notebook / tensors
- Code: numpy / matlibplot
- Run: Energy code / git
- Run: Fruit code
- Run: TicTacToe code
- Talk: Outro

#### Lessons

##### Course Intro

##### Introduce Energy Problem / numpy / matlibplot
##### Inference script and visualization
##### Install PyTorch and "Hello PyTorch!" / jupyter notebook
##### Fruit project
##### Tic-Tac-Toe project
##### Outro

#### Skills

- Use Tensors in PyTorch
- Git clone and Run basic training script
- Visualize with numpy and matlibplot
- Jupyter notebook

#### Vocab


- Training: Adjusting the parameters of a neural net from training data.
- Inference: Running a trained neural net to get an output for real.

- Linear or Fully Connected Layer: A trainable layer that connects every input to output with linear weights and a bias per output. Weights are a matrix.
- Activation Layer: Non-linear layer. Needed sandwiched between linear layers to model complex behaviour. ReLu is the cheapest and simplest.


- Scalar: An individual value. Think a floating point value.
- Vector: A bunch of values. Like x and y together form a position vector.
- Matrix: Drawn on paper in two dimensions. Could represent a set of equations or every combination of vectors. The topic of Linear Algebra has a lot to say about matrices. Modern AI hardware does large matrix multiplication at a blazing rate.
- Tensor: A higher level data structure that can be any of the above to the nth degree. A Rank 0 tensor is a scalar, rank 1 is a vector, rank 2 is a matrix, and so forth. The shape of a tensor describes it's rank and the dimensions at each level. Tensor is the data type in PyTorch that unifies all behaviour and allows data to be sent to a GPU. Think "Everything is a Tensor."

- Weights: A matrix that has values to multiply by every input to get every output.
- Biases: A vector that is an offset to add to each output of a linear layer.
- Parameters: The weights and biases of the all the linear layers of a model taken together.

- Loss Function (Cost Function): A function that takes the actual outputs (labels) and predicted outputs (from the forward pass) for each training sample and determines how far off it was. This is the starting point for adjusting the weights for each data point in training. CrossEntropyLoss is often used.
- Mean Squared Loss: Simple linear regression without taking the sqrt.
- Loss: The actual numerical value that is calculated using the loss function with the actual and predicted output vectors as arguments.

- Epoch: Running through all the training data once. 
- Accuracy: Percentage of the samples where the prediction is correct

- Logits: The probability scores output by a model. It can be thought of a one hot encoding of the classification or prediction. Usually this is then put through a final classifier layer.
- Softmax: The most common classifier. It works like an exponentially scaled probability. It's so effective because it still results in non-deterministic output, but scaled so that only alternatives that are almost as good are considered.

- Vanishing Gradiants:

- Zero Rule Baseline:
- One Rule Baseline:

#### Project 1a: Energy Demand Predictor

#### Project 1b: Fruit Classifier

#### Project 1c: Tic-Tac-Toe Solver

#### Bonus Project 1d: Calculator I

#### Presentation: Visualization of Inference

#### Sidenotes

- History of Neural Nets:

### Day 2: Training with Stochastic Gradient Descent

#### Skills

- Training Loop
- cuda
- SGD
- CrossEntropyLoss
- DataLoader

#### Vocab
- Gradients: The partial derivative of loss with respect to each weight. Think of the input, labeled output, and predicted output (calculated in the forward pass) as constant for each training sample, and each gradient represents how much changing a single weight would affect the loss (or improve the model for that particular sample).PyTorch (Autograd) is able to take the derivatives automatically from a Python implementation of the forward pass.
- Chain Rule: From multivariable calculus, it basically says that you can multiply the gradients from each layer to get  the next one. Or in other words, how much a change to the weight in a lower layer affects the final output (and loss) is the product of all those relationships through all the intermediate layers.
- Backpropagation: The process of recording the gradients at each layer and using them to calculate the gradients in the previous layer using the Chain Rule. PyTorch does the heavy lifting behind the scenes with just a few details in the training script.
- Learning Rate: The value that determines how much to adjust the weights during every pass. Often controlled by a more complex function with it's own hyperparameters.
- Hyperparameter: A value like learning rate that is tuned to improve the training process. Not a parameter in the model itself. Usually a global variable in the training script or in a .json config file.

- Stochastic: A system with randomness. See non-determinism and random variables.
- Gradient Descent: Using the backpropagated gradients, we actually adjust the weight in the opposite direction of the gradient to go in the direction (in n-dimensional space) "most downward" in the sense that it minimizes loss. Remember that the gradient represents how an increase in the weight would increase the loss, and we are trying to reduce the loss so we multiply the gradient by -1; or descent. The landscape that we are going down is specific to that data sample, but by choosing an appropriate learning rate and dropout, we can get to a stable (and good enough even if not the best) minimum where loss is low and accuracy is high for a real problem.
- Dropout: Randomly selecting only some of the weights when adjusting the weights during training.
- Local Minimum: For a multivariate function if we calculate the gradient and iteratively go down this, we'll hit a point where all the partial derivatives become zero; equalibrium. But there may be a point lower somewhere far away, and the most low of them all is the global minimum. But another wrinkle is that in practice we are iterating through different samples that all have totally different landscapes, so we always have to remember we are trying to find a stable local minimum that's good enough and that is not overfit to a single data point.

- Overfitting: A model that it overly specific to a narrow set of training data. High learning rates and small data sets lead to overfitting generally. Dropout increases robustness and reduces overfitting.


- Entropy: A measure of disorder of a system. Or think of it as the inverse of the ability to predict something about the system. See Gibb's Free Energy and Information Theory. 
- Cross Entropy Loss: Also known as log loss or logistic loss. In this case loss uses the ratio of entropy which invloves a log.


#### Notes
- 

#### Project 2a: 

#### Project 2b: 

#### Project 2c: 

#### Project 2d: 

#### Presentation: Visualization of SGD

#### Sidenotes

- History of Neural Nets:

### Day 3: Convolutional Neural Nets

#### Vocab

- Kernel


#### Project 3a: Image Classifier

#### Project 3b: Object Detection

#### Project 3c: Defect Detection

#### Presentation: Visualization of CNN

Topics: Transfer Learning

### Day 4: CNNs with Residual Connections / ML Hardware I

### Day 5: Embeddings / Recurrent Neural Nets

#### Project 5a: Embedding

#### Project 5b: Sentiment Analysis

#### Project 5c: Text Generation

#### Presentation: Visualization of RNN

### Day 6: LSTM / ML Hardware II / SGD mini-review

### Day 7: Transformers I

#### Project 7a: LLM Finetuning and Performance

#### Project 7b: LLM Inference and Performance

#### Project 7c: LLM Full Training on Novel Architecture

#### Presentation: Visualization of Transformer

### Day 8: Transformers II

#### Bonus Project 8a: Calculator II

### Day 9: Generative Adversarial Networks / Autoencoders

#### Vocab

- Latent Information Space

#### Project 9a: Image Generator

#### Presentation: Visualization of GAN

### Day 10: Stable Diffusion

#### Vocab

Denoising Network

#### Project 10a: Image Generator

#### Presentation: Visualization of LDM

### Day 11: Transformers III

Explainable AI

### Day 12: Recap and Review SGD

### Day 13: Catchup Day

Part II - Above the Model: Practicle High Level Techniques
----------------------------------------------------------


### Day 14: Reinforcement Learning I

### Day 15: Reinforcement Learning II

### Day 16: Transfer Learning / Fine Tuning

### Day 17: Few Shot Learning / Zero Shot Learning

### Day 18: Data Parallel / Model Parallel

### Day 19: Prompt Engineering

### Day 20: LangChain I

### Day 21: LangChain II

### Day 22: After Self-Attention / Start Capstone Projects

### Day 23: Multi-Modal / Augmented Transformer / Continue Capstone

### Day 24: Autonoumous Agents / Intelligence Discussion / Continue Capstone

### Day 25: Finish Capstone Projects / Recap


External Resources
------------------

### Part I
- Grokking Deep Learning
- Fast AI
- PyTorch tutorials

### Part II
- LangChain
- Papers