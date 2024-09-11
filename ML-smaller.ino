#define INPUTS 4
#define HIDDEN_NODES 4
#define OUTPUTS 1
#define DELTA 0.0001

// Example weights (initialize with your values or random values)

float weights1[INPUTS][HIDDEN_NODES] = {{0.1, -0.2, 0.3, -0.4},
{0.5, -0.6, 0.7, -0.8},
{0.9, -1.0, 1.1, -1.2},
{1.3, -1.4, 1.5, -1.6}};

float weights2[HIDDEN_NODES][OUTPUTS] = {{0.1}, {0.2}, {0.3}, {0.4}};
float biases1[HIDDEN_NODES] = {0.1, 0.1, 0.1, 0.1};
float bias2 = 0.1; // Bias for the output node
float inputs[INPUTS]; // Input values
float hidden[HIDDEN_NODES]; // Hidden layer activations
float output; // Output value

// ReLU activation function

float relu(float x) {
  return max(0, x);
}

// Forward pass

void forwardPass(float* inputs) {
  // Calculate hidden layer activations
  for (int j = 0; j < HIDDEN_NODES; j++) {
    hidden[j] = biases1[j];
    for (int i = 0; i < INPUTS; i++) {
      hidden[j] += inputs[i] * weights1[i][j];
    }
    hidden[j] = relu(hidden[j]);
  }

  // Calculate output
  output = bias2;
  for (int j = 0; j < HIDDEN_NODES; j++) {
    output += hidden[j] * weights2[j][0];
  }

  output = relu(output);
}

// Train to this:
float target = 0.6; // Example target value

// Backpropagation
void backpropagate(float* inputs, float learning_rate) {
  float error = output - target;
  float output_gradient = (output > 0) ? 1 : 0; // Derivative of ReLU is 1 for positive input, 0 otherwise
  output_gradient *= error; // Gradient for output node is error * derivative of activation function

  // Adjust bias for output layer
  bias2 -= learning_rate * output_gradient;

  // Adjust weights for output layer
  for (int j = 0; j < HIDDEN_NODES; j++) {
    weights2[j][0] -= learning_rate * hidden[j] * output_gradient;
  }

  // Backpropagate to hidden layers
  float hidden_gradient[HIDDEN_NODES];
  for (int j = 0; j < HIDDEN_NODES; j++) {
    hidden_gradient[j] = 0;
    if (hidden[j] > 0) { // Derivative of ReLU for hidden nodes
      hidden_gradient[j] = output_gradient * weights2[j][0];
    }
  }

  // Adjust biases and weights for hidden layer
  for (int j = 0; j < HIDDEN_NODES; j++) {
    biases1[j] -= learning_rate * hidden_gradient[j];
    for (int i = 0; i < INPUTS; i++) {
      weights1[i][j] -= learning_rate * inputs[i] * hidden_gradient[j];
    }
  }
}

void setup() {
  Serial.begin(115200);

  // inputs
  inputs[0] = 0.1;
  inputs[1] = 0.2;  
  inputs[2] = 0.3;
  inputs[3] = 0.4;

  // Fwd pass
  forwardPass(inputs);
  Serial.print("Output before training: ");
  Serial.println(output);

  // Perform backpropagation multiple times to simulate training
  for (int epoch = 0; epoch < 10000; epoch++) {
    forwardPass(inputs);
    backpropagate(inputs, DELTA);
  }

  // Forward pass again to see the adjusted output
  forwardPass(inputs);
  Serial.print("Output after training: ");
  Serial.println(output);
}

void loop() {
  // Nada
  delay(5);
}
