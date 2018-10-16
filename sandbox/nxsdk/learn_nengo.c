#include <stdlib.h>
#include <string.h>
#include "learn_nengo.h"

int guard_learn(runState *s) {
    return 1;
}

// Handles passing learning information to the correct learning rules
// to implement PES learning on Loihi.
//
// The required data is passed to this snip from the standard nengo_io
// snip via the userData structure. The data format is as follows:
//
//  0 :  n_errors
//    the number of learning signals. This is the same as the number
//    of Connections in the original Nengo model that terminate on
//    a conn.learning_rule.
//
//    This indicates how many copies of the following block there will be.
//  1 : core
//    The core id for the weights of the first learning connection
//  2 :  n_vals
//    The number of error signal dimensions.
//  3..3+n_vals : error_sig
//    The error signal, which has been multiplied by 100, rounded to an int,
//    and clipped to the [-100, 100] range.

void learn_nengo(runState *s) {
    printf("Doing Management Every Even cycles\n");
    int inputChannelID = getChannelID("inputChannel");
    /* int recvChannelID  = getChannelID("recvChannel"); */
    if (inputChannelID == -1) {
        printf("Invalid Channel ID\n");
        return;
    }

    int data[100];
    readChannel(inputChannelID, data, 1);
    readChannel(inputChannelID, data, 1);
    int n_errors = data[0];
    int core = data[1];
    if (n_errors > 100 | n_errors < 1) {
        printf("Invalid number of errors");
        return;
    }

    for (int i = 0; i < n_errors; i++) {
        readChannel(inputChannelID, data + i, 1);
    }

    int error, cx_idx;
    for (int i = 0; i < n_errors; i++) {
        error = (signed char) data[i];
        printf("Error %d: %d\n", i, error);
        neuron = NEURON_PTR((CoreId) {.id=core});
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
