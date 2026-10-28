import subprocess
import time


def run_command(command):
    print(f"\nRunning: {command}")
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    return result.returncode


def baseline_test():
    print("Starting baseline network test...")

    # Start Mininet
    print("\nStart Mininet manually with:")
    print(
        "sudo mn --custom network/topology.py "
        "--topo digitaltwin --switch ovsbr --controller=none"
    )

    print("\nInside Mininet, run:")
    print("h4 iperf3 -s &")
    print("h1 iperf3 -c 10.0.0.4 -t 10")
    print("h1 ping -c 20 h4")


if __name__ == "__main__":
    baseline_test()

