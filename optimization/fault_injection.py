import math
import subprocess


_INJECTED_INTERFACES = {}
_NO_QDISC_MESSAGES = (
    "no such file or directory",
    "cannot delete qdisc with handle of zero",
    "no such file",
)


def _get_link_interfaces(network, node_a, node_b):
    for link in getattr(network, "links", []):
        first, second = link.intf1, link.intf2
        if {first.node.name, second.node.name} == {node_a, node_b}:
            interfaces = {first.node.name: first, second.node.name: second}
            return interfaces[node_a], interfaces[node_b]
    raise ValueError(f"No Mininet link exists between {node_a} and {node_b}")


def _run_tc(interface, arguments, allow_missing=False):
    command = [
        "tc",
        *arguments[:2],
        "dev",
        interface.name,
        *arguments[2:],
    ]
    process = interface.node.popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError(f"tc timed out on {interface.name}")
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.communicate()
        raise

    if process.returncode == 0:
        return
    output = f"{stdout}\n{stderr}".strip()
    if allow_missing and any(message in output.lower() for message in _NO_QDISC_MESSAGES):
        return
    raise RuntimeError(f"tc failed on {interface.name}: {output}")


def _apply_fault(network, node_a, node_b, netem_arguments):
    interface_a, _ = _get_link_interfaces(network, node_a, node_b)
    _run_tc(
        interface_a,
        ["qdisc", "replace", "root", "netem", *netem_arguments],
    )
    _INJECTED_INTERFACES.setdefault(id(network), {})[
        frozenset((node_a, node_b))
    ] = (interface_a,)
    return interface_a.name


def add_latency(network, node_a, node_b, delay_ms):
    """Add one-way delay to packets leaving node_a on the selected link."""
    if isinstance(delay_ms, bool) or not isinstance(delay_ms, (int, float)):
        raise ValueError("delay_ms must be a non-negative number")
    if not math.isfinite(delay_ms) or delay_ms < 0:
        raise ValueError("delay_ms must be a finite non-negative number")
    interface = _apply_fault(
        network,
        node_a,
        node_b,
        ["delay", f"{delay_ms:g}ms"],
    )
    print(f"Added {delay_ms:g} ms egress delay on {interface} ({node_a} -> {node_b}).")
    return interface


def add_packet_loss(network, node_a, node_b, loss_percent):
    """Add one-way packet loss to packets leaving node_a on the selected link."""
    if isinstance(loss_percent, bool) or not isinstance(loss_percent, (int, float)):
        raise ValueError("loss_percent must be a number from 0 to 100")
    if not math.isfinite(loss_percent) or not 0 <= loss_percent <= 100:
        raise ValueError("loss_percent must be a finite number from 0 to 100")
    interface = _apply_fault(
        network,
        node_a,
        node_b,
        ["loss", f"{loss_percent:g}%"],
    )
    print(
        f"Added {loss_percent:g}% egress packet loss on {interface} "
        f"({node_a} -> {node_b})."
    )
    return interface


def remove_fault(network, node_a=None, node_b=None):
    """Remove injected qdiscs, optionally limited to a selected link."""
    if (node_a is None) != (node_b is None):
        raise ValueError("node_a and node_b must be provided together")

    injected = _INJECTED_INTERFACES.get(id(network), {})
    if node_a is None:
        interfaces = [
            interface
            for link_interfaces in injected.values()
            for interface in link_interfaces
        ]
    else:
        link_key = frozenset((node_a, node_b))
        if link_key in injected:
            interfaces = list(injected[link_key])
        else:
            interfaces = list(_get_link_interfaces(network, node_a, node_b))

    errors = []
    for interface in interfaces:
        try:
            _run_tc(interface, ["qdisc", "del", "root"], allow_missing=True)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError("Could not remove injected tc fault(s): " + "; ".join(errors))

    if node_a is None:
        _INJECTED_INTERFACES.pop(id(network), None)
    else:
        injected.pop(frozenset((node_a, node_b)), None)
    return len(interfaces)
