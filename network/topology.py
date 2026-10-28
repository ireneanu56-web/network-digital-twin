from mininet.topo import Topo


class DigitalTwinTopo(Topo):

    def build(self):

        # Hosts
        h1 = self.addHost("h1")
        h2 = self.addHost("h2")
        h3 = self.addHost("h3")
        h4 = self.addHost("h4")

        # Switches with STP enabled
        s1 = self.addSwitch("s1", stp=True)
        s2 = self.addSwitch("s2", stp=True)
        s3 = self.addSwitch("s3", stp=True)
        s4 = self.addSwitch("s4", stp=True)

        # Source
        self.addLink(h1, s1)

        # Two redundant paths
        self.addLink(s1, s2)
        self.addLink(s1, s3)

        # Converge at destination-side switch
        self.addLink(s2, s4)
        self.addLink(s3, s4)

        # Destination
        self.addLink(s4, h4)

        # Additional traffic hosts
        self.addLink(h2, s2)
        self.addLink(h3, s3)


topos = {
    "digitaltwin": lambda: DigitalTwinTopo()
}
