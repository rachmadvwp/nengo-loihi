import logging

from nengo import Network
from nengo.builder.builder import Builder as NengoBuilder
from nengo.builder.network import build_network


logger = logging.getLogger(__name__)


class Builder(NengoBuilder):
    """Fills in the Loihi Model object based on the Nengo Network.

    We cannot use the Nengo builder as is because we make normal Nengo
    networks for host-to-chip and chip-to-host communication. To keep
    Nengo and Nengo Loihi builders separate, we make a blank subclass,
    which effectively copies the class.
    """

    builders = {}


Builder.register(Network)(build_network)
