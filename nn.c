#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_INPUTS 2
#define N_HIDDEN 3
#define N_OUTPUTS 1
#define LEARNING_RATE 0.3
#define N_EPOCHS 10000


double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

double derivative_sigmoid(double x) {
  return x * (1.0 - x);
}

int main(int argc, char** argv) {
  srand(time(NULL));

  double inputs[4][N_INPUTS] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  double expected_outputs[4] = {0, 1, 1, 0};

  // Initialize weights and biases
  double w1[N_INPUTS][N_HIDDEN];
  double b1[N_HIDDEN];
  double w2[N_HIDDEN][N_OUTPUTS];
  double b2[N_OUTPUTS];
  for (int i = 0; i < N_INPUTS; i++) {
    for (int j = 0; j < N_HIDDEN; j++) {
      w1[i][j] = 2.0 * rand() / RAND_MAX - 1.0; // Random number from -1 to 1
    }
  }
  for (int i = 0; i < N_HIDDEN; i++) {
    b1[i] = 2.0 * rand() / RAND_MAX - 1.0;
  }
  for (int i = 0; i < N_HIDDEN; i++) {
    for (int j = 0; j < N_OUTPUTS; j++) {
      w2[i][j] = 2.0 * rand() / RAND_MAX - 1.0;
    }
  }
  for (int i = 0; i < N_OUTPUTS; i++) {
    b2[i] = 2.0 * rand() / RAND_MAX - 1.0;
  }

  for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
    // Loop through each input and expected output
    for (int i = 0; i < 4; i++) {
      double input[N_INPUTS] = {inputs[i][0], inputs[i][1]};
      double expected_output = expected_outputs[i];

      // Forward pass
      double z1[N_HIDDEN];
      double a1[N_HIDDEN];
      for (int j = 0; j < N_HIDDEN; j++) {
        z1[j] = 0;
        for (int k = 0; k < N_INPUTS; k++) {
          z1[j] += w1[k][j] * input[k];
        }
        z1[j] += b1[j];
        a1[j] = sigmoid(z1[j]);
      }
      double z2[N_OUTPUTS];
      double a2[N_OUTPUTS];
      for (int j = 0; j < N_OUTPUTS; j++) {
        z2[j] = 0;
        for (int k = 0; k < N_HIDDEN; k++) {
          z2[j] += w2[k][j] * a1[k];
        }
        z2[j] += b2[j];
        a2[j] = sigmoid(z2[j]);
      }

      // Backward pass
      double error = expected_output - a2[0];
      double delta2[N_OUTPUTS];
      for (int j = 0; j < N_OUTPUTS; j++) {
        delta2[j] = error * derivative_sigmoid(a2[j]);
      }
      double delta1[N_HIDDEN];
      for (int j = 0; j < N_HIDDEN; j++) {
        delta1[j] = 0;
        for (int k = 0; k < N_OUTPUTS; k++) {
          delta1[j] += w2[j][k] * delta2[k];
        }
        delta1[j] *= derivative_sigmoid(a1[j]);
      }

      // Update weights and biases
      for (int j = 0; j < N_HIDDEN; j++) {
        for (int k = 0; k < N_INPUTS; k++) {
          w1[k][j] += LEARNING_RATE * delta1[j] * input[k];
        }
        b1[j] += LEARNING_RATE * delta1[j];
      }
      for (int j = 0; j < N_OUTPUTS; j++) {
        for (int k = 0; k < N_HIDDEN; k++) {
          w2[k][j] += LEARNING_RATE * delta2[j] * a1[k];
        }
        b2[j] += LEARNING_RATE * delta2[j];
      }
    }
  }

  // Test the neural network
  for (int i = 0; i < 4; i++) {
    double input[N_INPUTS] = {inputs[i][0], inputs[i][1]};

    // Forward pass
    double z1[N_HIDDEN];
    double a1[N_HIDDEN];
    for (int j = 0; j < N_HIDDEN; j++) {
      z1[j] = 0;
      for (int k = 0; k < N_INPUTS; k++) {
        z1[j] += w1[k][j] * input[k];
      }
      z1[j] += b1[j];
      a1[j] = sigmoid(z1[j]);
    }
    double z2[N_OUTPUTS];
    double a2[N_OUTPUTS];
    for (int j = 0; j < N_OUTPUTS; j++) {
      z2[j] = 0;
      for (int k = 0; k < N_HIDDEN; k++) {
        z2[j] += w2[k][j] * a1[k];
      }
      z2[j] += b2[j];
      a2[j] = sigmoid(z2[j]);
    }

    printf("Input: [%d, %d], Output: %f, Expected: %f\n", (int)input[0], (int)input[1], a2[0], expected_outputs[i]);
  }

  return 0;
}