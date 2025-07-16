# Basics
## Supervised vs. unsupervised learning
### Formal version

**Supervised learning** involves learning a mapping from inputs XX to known outputs YY, using labeled data (Xi,Yi)(Xi​,Yi​). The model is explicitly trained to minimize prediction error on these labels.

**Unsupervised learning** involves finding patterns or structure in data XX without labeled outputs. The goal is typically to uncover latent structure, clusters, or representations.

### Metaphorical / vivid version:
Imagine giving a child a set of flashcards:

- In supervised learning, each card has both a picture (input) and the correct name (label). The child is being tutored—they get explicit feedback on whether their answers are right or wrong.
- In unsupervised learning, the child gets a stack of unlabeled pictures and is asked to sort them into meaningful piles. There’s no answer key—they’re left to figure out the structure on their own.

### EAS examples of both
#### Supervised: Precipitation prediction from satellite data

You have a dataset of satellite-derived variables (e.g., brightness temperatures, cloud top heights, humidity profiles) and co-located ground-based rain gauge measurements indicating precipitation rates.

    Inputs XX: Satellite-derived features

    Outputs YY: Observed rainfall rates

    Task: Learn a function f(X)≈Yf(X)≈Y

Framing: This is supervised learning because the model is explicitly trained to predict a known quantity (rainfall) from inputs.
Unsupervised Learning in EAS

#### Unsupervised: Classifying cloud regimes from satellite imagery

You have thousands of cloud snapshot images or pixel-level reflectance data, but no labels about cloud type or associated weather systems.

    Inputs XX: Multidimensional image data

    No outputs YY: No labels provided

    Task: Automatically group similar cloud types or identify dominant patterns

Framing: This is unsupervised learning because the goal is to uncover intrinsic structure (e.g., recurring cloud patterns or regimes) without any external labels.

## Train-validate-test split
- Good tool: `sklearn.train_test_split`
- k-folding cross validation

## Normalize the data: standardized anomalies

## Balancing the data

# Neural networks: key concepts

## neuron
A function that accepts one or more inputs and produces one single output. 

The **output** is determined by the values of the neuron's **inputs**, the **weights** that multiply each of those input values, the **bias** which scales the summed weighted inputs up or down, and the **activation function** which takes the weighted input (including the bias) and turns it into a scalar.

## layer
- input layer: the first layer, i.e. what the model takes in
- output layer: the final layer, i.e. what the model spits out
- hidden layer: any layer that's not the input layer or the output layer

## weight ($w_{ji}^l$)
For each neuron and each input to that neuron, a factor multiplying that input to control how strongly it influences that neuron's output.

Notation: 
- superscript ($l$) signifies which **layer**
- subscript $j$ signifies the **neuron** *in this layer*, layer $l$
- subscript $i$ siginfies the **neuron** *in the preceding layer*, layer $l-1$

Thus, the **weight** $w_{ji}^l$ is the number that the $j$th neuron in layer $l$ multiplies by the output of the $i$th neuron in layer $l-1$

## bias 
A scalar added to a given neuron's weighted inputs before passing them to the neuron's activation function.  Denoted $b_j^l$.

Unlike the weights, which are different for each input, the bias is a single number for each neuron.  So there is only one subscript: $b_j^l$ is the bias for the $j$th neuron in layer $l$.

## weighted input
The inputs multiplied by their weights, plus the bias. In vector form for all the neurons in a given layer: $z^l=w^la^{l-1}+b^l$

## activation function 
The function $f_j^l$ that takes a neuron's weighted inputs and produces a single, scalar output.  Examples include sigmoid, ReLU, many others.

## activation 
The neuron's output, denoted $a_j^l$: simply the output of the neuron's **activation function**, given the neuron's **weighted input**: $a_j^l=f_j^l(z_j^l)$

## cost function
The function, $C$, quantifying the model's accuracy.  Examples include quadratic, cross-entropy, many others.

## error
The derivative of the cost function with respect to a given neuron's weighted input: $\delta_j^l=\partial C/\partial z_j^l$

## gradient descent
An algorithm that uses the training data to iteratively update the model's weights and biases so as to minimize the cost function. 

Specifically, it uses **backpropagation** to compute the gradient of the model's cost function, $\nabla C$, with respect to all of the model's weights and biases: $$\nabla C=\left(\frac{\partial C}{\partial b_1^1},\frac{\partial C}{\partial b_2^1},\cdots,\frac{\partial C}{\partial w_{I,J-1}^L},\frac{\partial C}{\partial w_{IJ}^L}\right)$$

This involves the **error** quantity defined above.

## stochastic gradient descent
A version of **gradient descent** that uses only a subset of the training data at a time, and then averages across those "mini batches"