#include <stdlib.h>
#include <string.h>
#include "learn_nengo.h"

int guard_learn(runState *s) {
    return 1;
}

void learn_nengo(runState *s) {
    int n_errors = s->userData[0];
    int core = s->userData[1];
    printf("Learn: n_errors=%d, core=%d\n", n_errors, core);
    NeuronCore *neuron = NEURON_PTR((CoreId) {.id=core});

    int error, cx_idx;
    for (int i = 0; i < n_errors; i++) {
        error = (signed char) s->userData[i+2];
        printf("Error %d: %d\n", i, error);

        cx_idx = i;

        if (error > 0) {
            neuron->stdp_post_state[cx_idx] = \
                (PostTraceEntry) {
                    .Yspike0      = 0,
                    .Yspike1      = 0,
                    .Yspike2      = 0,
                    .Yepoch0      = abs(error),
                    .Yepoch1      = 0,
                    .Yepoch2      = 0,
                    .Tspike       = 0,
                    .TraceProfile = 3,
                    .StdpProfile  = 1
                };
            neuron->stdp_post_state[cx_idx+n_errors] = \
                (PostTraceEntry) {
                    .Yspike0      = 0,
                    .Yspike1      = 0,
                    .Yspike2      = 0,
                    .Yepoch0      = abs(error),
                    .Yepoch1      = 0,
                    .Yepoch2      = 0,
                    .Tspike       = 0,
                    .TraceProfile = 3,
                    .StdpProfile  = 0
                };
        } else {
            neuron->stdp_post_state[cx_idx] = \
                (PostTraceEntry) {
                    .Yspike0      = 0,
                    .Yspike1      = 0,
                    .Yspike2      = 0,
                    .Yepoch0      = abs(error),
                    .Yepoch1      = 0,
                    .Yepoch2      = 0,
                    .Tspike       = 0,
                    .TraceProfile = 3,
                    .StdpProfile  = 0
                };
            neuron->stdp_post_state[cx_idx+n_errors] = \
                (PostTraceEntry) {
                    .Yspike0      = 0,
                    .Yspike1      = 0,
                    .Yspike2      = 0,
                    .Yepoch0      = abs(error),
                    .Yepoch1      = 0,
                    .Yepoch2      = 0,
                    .Tspike       = 0,
                    .TraceProfile = 3,
                    .StdpProfile  = 1
                };
        }
    }
}
