/*
 * Reinforcement Learning Book / Exercise 4.7
 *
 * $ sudo apt-get install libgsl-dev libgsl2 
 * $ gcc car_rentals.c -lgsl -lgslcblas -lm -o car_rentals
 */


#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_randist.h>

#define Size 20
#define Iteration 100
#define Rent1_mean 3
#define Rent2_mean 4
#define Return1_mean 3
#define Return2_mean 2

// Global variables
float value[Size + 1][Size + 1];
int policy[Size + 1][Size + 1];


void show_value(){
    printf("[");
    for (int c1 = 0; c1 <= Size; c1++) {
        printf("[");
        for (int c2 = 0; c2 <= Size; c2++) {
            printf("%3.0f", value[c1][c2]);
            if (c2 < Size) {
                printf(", ");
            }
        }
        printf("]");
        if (c1 < Size) {
            printf(",\n ");
        }
    }
    printf("]\n");
}


void show_policy(){
    printf("[");
    for (int c1 = 0; c1 <= Size; c1++) {
        printf("[");
        for (int c2 = 0; c2 <= Size; c2++) {
            printf("%2d", policy[c1][c2]);
            if (c2 < Size) {
                printf(", ");
            }
        }
        printf("]");
        if (c1 < Size) {
            printf(",\n ");
        }
    }
    printf("]\n");
}


float calc_q_val(int c1, int c2, int a){
    float q_val = 0;
    int _c1, _c2;
    float c1_rent_prob, c2_rent_prob, c1_return_prob, c2_return_prob, prob;

    // Moving cars
    c1 -= a;
    c2 += a;
    if (a > 0) {
        q_val -= 2*(a-1);
    } else {
        q_val -= 2*abs(a);
    }
    if (c1 > 10) q_val -= 4;
    if (c2 > 10) q_val -= 4;

    for (int c1_rent = 0; c1_rent <= c1; c1_rent++) {
        for (int c2_rent = 0; c2_rent <= c2; c2_rent++) {
            for (int c1_return = 0; c1_return <= Size-(c1-c1_rent); c1_return++) {
                for (int c2_return = 0; c2_return <= Size-(c2-c2_rent); c2_return++) {
                    _c1 = c1 - c1_rent + c1_return;
                    _c2 = c2 - c2_rent + c2_return;

                    if (c1_rent == c1) {
                        c1_rent_prob = 1;
                        for (int n = 0; n < c1_rent; n++) {
                            c1_rent_prob -= gsl_ran_poisson_pdf(n, Rent1_mean);
                        }
                    } else {
                        c1_rent_prob = gsl_ran_poisson_pdf(c1_rent, Rent1_mean);
                    }

                    if (c2_rent == c2) {
                        c2_rent_prob = 1;
                        for (int n = 0; n < c2_rent; n++) {
                            c2_rent_prob -= gsl_ran_poisson_pdf(n, Rent2_mean);
                        }
                    } else {
                        c2_rent_prob = gsl_ran_poisson_pdf(c2_rent, Rent2_mean);
                    }

                    if (c1_return == Size-(c1-c1_rent)) {
                        c1_return_prob = 1;
                        for (int n = 0; n < c1_return; n++) {
                            c1_return_prob -= gsl_ran_poisson_pdf(n, Return1_mean);
                        }
                    } else {
                        c1_return_prob = gsl_ran_poisson_pdf(c1_return, Return1_mean);
                    }

                    if (c2_return == Size-(c2-c2_rent)) {
                        c2_return_prob = 1;
                        for (int n = 0; n < c2_return; n++) {
                            c2_return_prob -= gsl_ran_poisson_pdf(n, Return2_mean);
                        }
                    } else {
                        c2_return_prob = gsl_ran_poisson_pdf(c2_return, Return2_mean);
                    }

                    prob = c1_rent_prob * c2_rent_prob * c1_return_prob * c2_return_prob;
                    q_val += prob * (10 * (c1_rent + c2_rent) + 0.9 * value[_c1][_c2]);
                }
            }
        }
    }
    return q_val;
}


int min(int a, int b){
    if (a < b) {
        return a;
    } else {
        return b;
    }
}


void run() {
    float q_val;
    float value_update;
    int policy_update;

    for (int c1 = 0; c1 <= Size; c1++) {
        for (int c2 = 0; c2 <= Size; c2++) {
            value[c1][c2] = 0;
            policy[c1][c2] = 0;
        }
    }

    printf("# Iteration 0\n");
    show_policy();
    show_value();

    for (int i = 1; i < Iteration; i++) {
        for (int c1 = 0; c1 <= Size; c1++) {
            for (int c2 = 0; c2 <= Size; c2++) {
                int update = 0;
                for (int a = -min(c2, 5); a <= min(c1, 5); a++) {
                    q_val = calc_q_val(c1, c2, a);
                    if (update == 0 || q_val > value_update) {
                        value_update = q_val;
                        policy_update = a;
                        update = 1;
                    }
                }
                value[c1][c2] = value_update;
                policy[c1][c2] = policy_update;
            }
        }
        printf("# Iteration %d\n", i);
        show_policy();
        show_value();
    }
}


int main() {
    run();
}