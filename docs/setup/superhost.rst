*********
Superhost
*********

Connecting to a host
====================

The host and superhost communicate through
a hardwired Ethernet connection.
The superhost therefore must have
at least two networks interfaces,
one for an external internet connection
and one to connect to the FPGA host.

The host only has one network interface,
which is connected to the superhost.
In order to access the internet,
the superhost must share
its external connection with the host.

To do this, assuming that you are running Ubuntu:

1. Open "Network Connections".

2. Identify the Ethernet connection being used
   to connect to the Loihi system.
   Clicking the network icon in the task bar
   will inform you which network interfaces are available.

3. Select "Wired connection <x>" and click "Edit".

4. Navigate to "IPv4 Settings" and change
   "Method" to "Shared to other computers".

5. Click "Save".

6. Check that the network interface has been assigned the correct IP.

   When the Ethernet cable between the host and superhost is connected, do:

   .. code-block:: bash

      sudo ifconfig -a

   to display the information for each network interface.
   The network interface being used to connect to the Loihi system
   should be assigned the IP ``10.42.0.1``.
