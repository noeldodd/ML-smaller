#define INPUTS 4
#define HIDDEN_NODES 4
#define OUTPUTS 1
#define DELTA 0.0001  // Fake calculus

// Times of day; makes fake training data easier to look at.
#define T_0000 0.0000  // 00:00 = 000/255
#define T_0100 0.0392  // 01:00 = 010/255
#define T_0200 0.0784  // 02:00 = 020/255
#define T_0300 0.1176  // 03:00 = 030/255
#define T_0400 0.1569  // 04:00 = 040/255
#define T_0500 0.1961  // 05:00 = 050/255
#define T_0600 0.2353  // 06:00 = 060/255
#define T_0700 0.2745  // 07:00 = 070/255
#define T_0800 0.3137  // 08:00 = 080/255
#define T_0900 0.3529  // 09:00 = 090/255
#define T_1000 0.3922  // 10:00 = 100/255
#define T_1100 0.4314  // 11:00 = 110/255
#define T_1200 0.4706  // 12:00 = 120/255
#define T_1300 0.5098  // 13:00 = 130/255
#define T_1400 0.5490  // 14:00 = 140/255
#define T_1500 0.5882  // 15:00 = 150/255
#define T_1600 0.6275  // 16:00 = 160/255
#define T_1700 0.6667  // 17:00 = 170/255
#define T_1800 0.7059  // 18:00 = 180/255
#define T_1900 0.7451  // 19:00 = 190/255
#define T_2000 0.7843  // 20:00 = 200/255
#define T_2100 0.8235  // 21:00 = 210/255
#define T_2200 0.8627  // 22:00 = 220/255
#define T_2300 0.9020  // 23:00 = 230/255

// Same for Day of Week
#define D_MONDAY           0.0    // 0: Monday
#define D_TUESDAY          0.1216 // 1: Tuesday
#define D_WEDNESDAY        0.2471 // 2: Wednesday
#define D_THURSDAY         0.3725 // 3: Thursday
#define D_FRIDAY           0.4980 // 4: Friday
#define D_SATURDAY         0.6235 // 5: Saturday
#define D_SUNDAY           0.7490 // 6: Sunday
#define D_STAT             0.8745 // 7: Placeholder weekdays that fall on a stat

// Example weights (initialize with realistic values or random values)
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
