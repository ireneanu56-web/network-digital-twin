from pathlib import Path
import subprocess
import time

import networkx as nx


_TOPOLOGY_EDGES = (
    ("h1", "s1"),
    ("s1", "s2"),
    ("s1", "s3"),
    ("s2", "s4"),
    ("s3", "s4"),
    ("s4", "h4"),
    ("h2", "s2"),
    ("h3", "s3"),
)
_CURRENT_PATHS = {}
_OPENFLOW_PRIORITY = 200


def build_network_graph(network=None):
    """Return a graph of the existing Mininet topology or a live Mininet net."""
    graph = nx.Graph()
    if network is None:
        graph.add_nodes_from(
            [
                ("h1", {"kind": "host"}),
                ("h2", {"kind": "host"}),
                ("h3", {"kind": "host"}),
                ("h4", {"kind": "host"}),
                ("s1", {"kind": "switch"}),
                ("s2", {"kind": "switch"}),
                ("s3", {"kind": "switch"}),
                ("s4", {"kind": "switch"}),
            ]
        )
        graph.add_edges_from(
            (left, right, {"cost": 1.0}) for left, right in _TOPOLOGY_EDGES
        )
        return graph

    nodes = getattr(network, "hosts", []) + getattr(network, "switches", [])
    for node in nodes:
        kind = "switch" if hasattr(node, "ports") and node in network.switches else "host"
        graph.add_node(node.name, kind=kind)
    for link in getattr(network, "links", []):
        left = link.intf1.node.name
        right = link.intf2.node.name
        graph.add_edge(left, right, cost=1.0)
    return graph


def _normalize_link(failed_link):
    if failed_link is None or len(failed_link) != 2:
        raise ValueError("failed_link must contain exactly two node names")
    return tuple(failed_link)


def get_current_path(src, dst, network=None):
    """Get the most recently installed path, or the graph's current shortest path."""
    key = (src, dst)
    if key in _CURRENT_PATHS:
        return list(_CURRENT_PATHS[key])
    graph = build_network_graph(network)
    try:
        return nx.shortest_path(graph, src, dst, weight="cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def find_alternate_path(src, dst, failed_link=None, graph=None):
    """Find a shortest route after excluding the faulty link."""
    candidate_graph = (graph or build_network_graph()).copy()
    if failed_link is not None:
        left, right = _normalize_link(failed_link)
        if candidate_graph.has_edge(left, right):
            candidate_graph.remove_edge(left, right)
    try:
        return nx.shortest_path(candidate_graph, src, dst, weight="cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def calculate_path_cost(path, graph=None):
    if path is None or len(path) < 2:
        raise ValueError("path must contain at least two connected nodes")
    graph = graph or build_network_graph()
    cost = 0.0
    for left, right in zip(path, path[1:]):
        if not graph.has_edge(left, right):
            raise ValueError(f"No topology link exists between {left} and {right}")
        cost += float(graph[left][right].get("cost", 1.0))
    return cost


def _link_interfaces(network, node_a, node_b):
    for link in getattr(network, "links", []):
        first, second = link.intf1, link.intf2
        if {first.node.name, second.node.name} == {node_a, node_b}:
            interfaces = {first.node.name: first, second.node.name: second}
            return interfaces[node_a], interfaces[node_b]
    raise ValueError(f"No live Mininet link exists between {node_a} and {node_b}")


def _interface_port(switch, interface):
    try:
        return int(switch.ports[interface])
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Could not determine OVS port for {interface.name} on {switch.name}"
        ) from error


def _run_ovs(arguments):
    return subprocess.run(
        ["ovs-ofctl", *arguments],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )


def _delete_host_pair_flows(network, source_ip, destination_ip):
    failures = []
    for switch in network.switches:
        for src_ip, dst_ip in (
            (source_ip, destination_ip),
            (destination_ip, source_ip),
        ):
            try:
                _run_ovs(
                    [
                        "del-flows",
                        switch.name,
                        f"ip,nw_src={src_ip},nw_dst={dst_ip}",
                    ]
                )
            except (OSError, subprocess.SubprocessError) as error:
                failures.append(f"{switch.name}: {error}")
    if failures:
        raise RuntimeError("Could not remove existing path flows: " + "; ".join(failures))


def _install_direction(network, path, source_ip, destination_ip):
    for previous, switch_name, following in zip(path, path[1:], path[2:]):
        switch = network.get(switch_name)
        _, ingress_interface = _link_interfaces(network, previous, switch_name)
        egress_interface, _ = _link_interfaces(network, switch_name, following)
        ingress_port = _interface_port(switch, ingress_interface)
        egress_port = _interface_port(switch, egress_interface)
        flow = (
            f"priority={_OPENFLOW_PRIORITY},ip,in_port={ingress_port},"
            f"nw_src={source_ip},nw_dst={destination_ip},actions=output:{egress_port}"
        )
        _run_ovs(["add-flow", switch.name, flow])


def reroute_traffic(network, src, dst, path):
    """Install high-priority OVS flows for a path in both traffic directions."""
    if not path or path[0] != src or path[-1] != dst:
        raise ValueError("path endpoints must match src and dst")
    graph = build_network_graph(network)
    for left, right in zip(path, path[1:]):
        if not graph.has_edge(left, right):
            raise ValueError(f"No live link exists between {left} and {right}")

    source_ip = network.get(src).IP()
    destination_ip = network.get(dst).IP()
    if not source_ip or not destination_ip:
        raise RuntimeError(f"Could not get IP addresses for {src} and {dst}")

    _delete_host_pair_flows(network, source_ip, destination_ip)
    try:
        _install_direction(network, path, source_ip, destination_ip)
        _install_direction(
            network,
            list(reversed(path)),
            destination_ip,
            source_ip,
        )
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
        try:
            _delete_host_pair_flows(network, source_ip, destination_ip)
        except RuntimeError as cleanup_error:
            raise RuntimeError(
                f"Failed to install OVS path: {error}; cleanup also failed: "
                f"{cleanup_error}"
            ) from error
        raise RuntimeError(f"Failed to install OVS path: {error}") from error

    _CURRENT_PATHS[(src, dst)] = list(path)
    _CURRENT_PATHS[(dst, src)] = list(reversed(path))
    return list(path)


def clear_route(network, src, dst):
    """Remove only the host-pair routes installed by this MVP."""
    source_ip = network.get(src).IP()
    destination_ip = network.get(dst).IP()
    _delete_host_pair_flows(network, source_ip, destination_ip)
    _CURRENT_PATHS.pop((src, dst), None)
    _CURRENT_PATHS.pop((dst, src), None)


def self_heal(network, src, dst, failed_link):
    """Exclude the reported faulty link and install the best live alternate route."""
    started = time.perf_counter()
    try:
        original_path = get_current_path(src, dst, network)
        if original_path is None:
            return {
                "original_path": None,
                "alternate_path": None,
                "decision_time": time.perf_counter() - started,
                "success": False,
                "reason": f"No current path exists from {src} to {dst}",
            }

        graph = build_network_graph(network)
        failed_left, failed_right = _normalize_link(failed_link)
        if not graph.has_edge(failed_left, failed_right):
            return {
                "original_path": original_path,
                "alternate_path": None,
                "decision_time": time.perf_counter() - started,
                "success": False,
                "reason": f"Faulty link {failed_left}-{failed_right} is not in the live topology",
            }

        alternate_path = find_alternate_path(
            src,
            dst,
            failed_link=(failed_left, failed_right),
            graph=graph,
        )
        if alternate_path is None:
            return {
                "original_path": original_path,
                "alternate_path": None,
                "decision_time": time.perf_counter() - started,
                "success": False,
                "reason": f"No alternate path avoids {failed_left}-{failed_right}",
            }

        reroute_traffic(network, src, dst, alternate_path)
        return {
            "original_path": original_path,
            "alternate_path": alternate_path,
            "decision_time": time.perf_counter() - started,
            "success": True,
            "reason": None,
        }
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        return {
            "original_path": _CURRENT_PATHS.get((src, dst)),
            "alternate_path": None,
            "decision_time": time.perf_counter() - started,
            "success": False,
            "reason": str(error),
        }
