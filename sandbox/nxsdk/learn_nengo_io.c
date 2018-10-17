#include <stdlib.h>
#include <string.h>
#include "learn_nengo_io.h"

int guard_learn_nengo_io(runState *s) {
    return 1;
}

void learn_nengo_io(runState *s) {
    int inputChannelID = getChannelID("inputChannel");
    int outputChannelID = getChannelID("outputChannel");
    if (inputChannelID == -1 || outputChannelID == -1) {
        printf("Invalid Channel ID\n");
        return;
    }

    int data[1];
    readChannel(inputChannelID, data, 1);
    int n_errors = data[0];
    readChannel(inputChannelID, data, 1);
    int core = data[0];
    printf("IO: n_errors=%d, core=%d\n", n_errors, core);

    if (n_errors > 100 | n_errors < 1) {
        printf("Invalid number of errors");
        return;
    }

    s->userData[0] = n_errors;
    s->userData[1] = core;
    for (int i = 0; i < n_errors; i++) {
        readChannel(inputChannelID, data, 1);
	s->userData[i+2] = data[0];
	printf("Error %d: %d\n", i, data[0]);
    }

    writeChannel(outputChannelID, 7, 1);
}
