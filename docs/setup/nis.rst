**************************************
NIS (User accounts and groups syncing)
**************************************

Adding a new user
=================

1. Log in as a user who can use ``sudo``.

2. Change to the super user.

   .. code-block:: bash

      sudo -s

3. Add the user.

   .. code-block:: bash

      adduser <username>

4. *(Optional)*: Enable the user to use ``sudo``.

   .. code-block:: bash

      usermod -aG sudo <username>

5. Add the user to the ``loihi_sudo`` group.

   This is necessary for allowing the user
   to run models on Loihi boards.

   .. code-block:: bash

      usermod -aG loihi_sudo <username>

6. Propagate the new user information to connected Loihi boards.

   .. code-block:: bash

      make -C /var/yp

You can then run ``exit`` to exit the superuser session.

Note that the final step copies user information
to the Loihi boards.
You therefore do not have to make a new user account
on the hosts or boards that are connected to the superhost.

To be sure that the user information has been copied correctly,
once finishing the above steps,
you should test by logging into all connected hosts and boards.

For example, on the superhost try

.. code-block:: bash

   ssh <username>@host-1
   ssh <username>@board-1
