#include <Arduino.h>

#define INPUTS 4
#define HIDDEN_LAYERS 4  // Two hidden layers
#define NODES_PER_LAYER 4  // Four nodes per hidden layer
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

const PROGMEM float training[][4] = {
  {D_MONDAY, T_0600, 180, 125},
  {D_TUESDAY, T_0600, 181, 125},
  {D_WEDNESDAY, T_0600, 180, 125},
  {D_THURSDAY, T_0600, 179, 125},
  {D_FRIDAY, T_0600, 182, 124},
  {D_SATURDAY, T_0800, 181, 124},
  {D_SUNDAY, T_0800, 178, 125},
  {D_STAT, T_0900, 176, 126},
  {D_MONDAY, T_0900, 180, 120},
  {D_TUESDAY, T_0900, 181, 120},
  {D_WEDNESDAY, T_0900, 180, 120},
  {D_THURSDAY, T_0900, 179, 120},
  {D_FRIDAY, T_0900, 182, 118},
  {D_SATURDAY, T_1200, 181, 119},
  {D_SUNDAY, T_1100, 175, 126},
  {D_STAT, T_1200, 174, 127},
  {D_MONDAY, T_1700, 183, 119},
  {D_TUESDAY, T_1700, 184, 118},
  {D_WEDNESDAY, T_1700, 184, 119},
  {D_THURSDAY, T_1700, 180, 119},
  {D_FRIDAY, T_1700, 170, 125},
  {D_SATURDAY, T_1800, 180, 122},
  {D_SUNDAY, T_1900, 181, 122},
  {D_STAT, T_2000, 170, 124},
  {D_MONDAY, T_2000, 180, 117},
  {D_TUESDAY, T_2000, 183, 115},
  {D_WEDNESDAY, T_2000, 182, 116},
  {D_THURSDAY, T_2000, 178, 119},
  {D_FRIDAY, T_2300, 182, 116},
  {D_SATURDAY, T_2200, 187, 112},
  {D_SUNDAY, T_2000, 185, 118},
  {D_STAT, T_2000, 170, 120}
};

// Weights and biases for the network
float weights1[INPUTS][NODES_PER_LAYER];
float weights_hidden[HIDDEN_LAYERS-1][NODES_PER_LAYER][NODES_PER_LAYER];
float weights2[NODES_PER_LAYER][OUTPUTS];
float biases1[NODES_PER_LAYER];
float biases_hidden[HIDDEN_LAYERS-1][NODES_PER_LAYER];
float bias2;
float inputs[INPUTS];
float hidden[HIDDEN_LAYERS][NODES_PER_LAYER];
float output;

#define OUTDOOR_MIN_TEMP -50.0f
#define OUTDOOR_SCALE_FACTOR 2.5f  //Outdoor scaling factor

#define INDOOR_MIN_TEMP -5.0f
#define INDOOR_SCALE_FACTOR 5.0f //Indoor scaling factor

// Outdoor temperature conversion (from °C to byte)
uint8_t outdoorTempToByte(float temperature) {
  if (temperature < -50.0f) temperature = -50.0f;
  if (temperature > 52.0f) temperature = 52.0f;

  return (uint8_t)((temperature + 50) * OUTDOOR_SCALE_FACTOR);
}

// Indoor temperature conversion (from °C to byte)
uint8_t indoorTempToByte(float temperature) {
  if (temperature < INDOOR_MIN_TEMP) temperature = INDOOR_MIN_TEMP;
  if (temperature > 45.0f) temperature = 45.0f;

  return (uint8_t)((temperature + 5) * INDOOR_SCALE_FACTOR);
}

// Reverse conversion
float byteToOutdoorTemp(uint8_t byteValue) {
  return ((float)byteValue / OUTDOOR_SCALE_FACTOR - 50);
}

float byteToIndoorTemp(uint8_t byteValue) {
  return ((float)byteValue / INDOOR_SCALE_FACTOR - 5);
}

// ReLU activation function
float relu(float x) {
  return max(0.0, x);
}

// Forward pass through the network
void forwardPass(float* inputs) {
  // Input layer to first hidden layer
  for (int j = 0; j < NODES_PER_LAYER; j++) {
    hidden[0][j] = biases1[j];
    for (int i = 0; i < INPUTS; i++) {
      hidden[0][j] += inputs[i] * weights1[i][j];
    }
    hidden[0][j] = relu(hidden[0][j]);
  }

  // Hidden layer to hidden layer
  for (int l = 1; l < HIDDEN_LAYERS; l++) {
    for (int j = 0; j < NODES_PER_LAYER; j++) {
      hidden[l][j] = biases_hidden[l-1][j];
      for (int i = 0; i < NODES_PER_LAYER; i++) {
        hidden[l][j] += hidden[l-1][i] * weights_hidden[l-1][i][j];
      }
      hidden[l][j] = relu(hidden[l][j]);
    }
  }

  // Last hidden layer to output layer
  output = bias2;
  for (int j = 0; j < NODES_PER_LAYER; j++) {
    output += hidden[HIDDEN_LAYERS-1][j] * weights2[j][0];
  }
  output = relu(output);
}

// Target value
float target = 0.0;

void backpropagate(float* inputs, float learning_rate) {
  float error = output - target;
  float output_gradient = (output > 0) ? 1 : 0;
  output_gradient *= error;

  // Adjust bias for output layer
  bias2 -= learning_rate * output_gradient;

  // Adjust weights for output layer
  float hidden_gradient[HIDDEN_LAYERS][NODES_PER_LAYER] = {0};
  for (int j = 0; j < NODES_PER_LAYER; j++) {
    weights2[j][0] -= learning_rate * hidden[HIDDEN_LAYERS-1][j] * output_gradient;

    if (hidden[HIDDEN_LAYERS-1][j] > 0) {
      hidden_gradient[HIDDEN_LAYERS-1][j] = output_gradient * weights2[j][0];
    }
  }

  // Backpropagate through hidden layers
  for (int l = HIDDEN_LAYERS-2; l >= 0; l--) {
    for (int j = 0; j < NODES_PER_LAYER; j++) {
      if (hidden[l][j] > 0) {
        hidden_gradient[l][j] = 0;
        for (int k = 0; k < NODES_PER_LAYER; k++) {
          hidden_gradient[l][j] += hidden_gradient[l+1][k] * weights_hidden[l][j][k];
        }
      }
    }
  }

  // Adjust biases and weights for each hidden layer
  for (int l = HIDDEN_LAYERS-1; l >= 0; l--) {
    for (int j = 0; j < NODES_PER_LAYER; j++) {
      if (l == 0) {
        biases1[j] -= learning_rate * hidden_gradient[l][j];
        for (int i = 0; i < INPUTS; i++) {
          weights1[i][j] -= learning_rate * inputs[i] * hidden_gradient[l][j];
        }
      } else {
        biases_hidden[l-1][j] -= learning_rate * hidden_gradient[l][j];
        for (int i = 0; i < NODES_PER_LAYER; i++) {
          weights_hidden[l-1][i][j] -= learning_rate * hidden[l-1][i] * hidden_gradient[l][j];
        }
      }
    }
  }
}

void setInputs(int index) {
  inputs[0] = pgm_read_float(&training[index][0]);
  inputs[1] = pgm_read_float(&training[index][1]);
  inputs[2] = pgm_read_float(&training[index][2])/255;
  inputs[3] = pgm_read_float(&training[index][3])/255;
}

#define SAMPLE_IDX 1
#define PER_SAMPLE_ROUNDS 8000

void setup() {
  Serial.begin(115200);

  // Initialize weights and biases with random values or zeros
  for (int i = 0; i < INPUTS; i++) {
    for (int j = 0; j < NODES_PER_LAYER; j++) {
      weights1[i][j] = random(100) / 100.0;
    }
  }

  for (int l = 0; l < HIDDEN_LAYERS-1; l++) {
    for (int i = 0; i < NODES_PER_LAYER; i++) {
      for (int j = 0; j < NODES_PER_LAYER; j++) {
        weights_hidden[l][i][j] = random(100) / 100.0;
      }
    }
  }

  for (int j = 0; j < NODES_PER_LAYER; j++) {
    weights2[j][0] = random(100) / 100.0;
    biases1[j] = random(100) / 100.0;
    bias2 = random(100) / 100.0;
  }

  for (int l = 0; l < HIDDEN_LAYERS-1; l++) {
    for (int j = 0; j < NODES_PER_LAYER; j++) {
      biases_hidden[l][j] = random(100) / 100.0;
    }
  }

  inputs[0] = pgm_read_float(&training[SAMPLE_IDX][0]);
  inputs[1] = pgm_read_float(&training[SAMPLE_IDX][1]);
  inputs[2] = pgm_read_float(&training[SAMPLE_IDX][2])/255;
  inputs[3] = pgm_read_float(&training[SAMPLE_IDX][3])/255;

  forwardPass(inputs);
  Serial.print("Output before training: ");
  Serial.print(byteToIndoorTemp(output));
  Serial.println("°C");

  for (int training_index = 0; training_index < 32; training_index++) {
    float target_raw = pgm_read_float(&training[training_index][3]);
    target = target_raw/255;
    Serial.print("Training pass # ");
    Serial.print(training_index);
    Serial.print(" target value is: ");
    Serial.print(byteToIndoorTemp(target_raw));
    Serial.println("°C");
    for (int epoch = 0; epoch < PER_SAMPLE_ROUNDS; epoch++) {
      setInputs(training_index);
      forwardPass(inputs);
      backpropagate(inputs, DELTA);
    }
  }

  for (int training_index = 0; training_index < 32; training_index++) {
    setInputs(training_index);
    forwardPass(inputs);
    Serial.print("Pass #");
    Serial.print(training_index);
    Serial.print(" Target output from training data: ");
    Serial.print(byteToIndoorTemp(inputs[3] * 255));
    Serial.print("°C");
    Serial.print(" Inf output (raw) after training: ");
    Serial.print(output);
    Serial.print(" or ");
    Serial.print(byteToIndoorTemp(output * 255));
    Serial.println("°C");
  }
}

void loop() {
  delay(5);
}