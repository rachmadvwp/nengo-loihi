#include <stdlib.h>
#include <string.h>
#include "nengo_io.h"

void nengo_io(runState *s) {
    int count[1];
    readChannel("nengo_io_h2c_1", count, 1);

    int spike[2];
    for (int i=0; i<count[0]; i++) {
        readChannel("nengo_io_h2c_1", spike, 2);
        CoreId coreId = (CoreId) {.id=spike[1]};
        nx_send_discrete_spike(s->time, coreId, spike[2]);
        //printf("spike: t=%d core=%d axon=%d\n", s->time, spike[0], spike[1]);
    }

    int output[1];
    for(int i=0; i<1; i++) {
        output[i] = s->time;
        writeChannel("nengo_io_c2h_1", output+i, 1);
    }
}
