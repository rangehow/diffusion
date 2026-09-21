"""
Pre-launch TCP barrier: wait until ALL nodes are alive before starting training.

Each node opens a TCP listener to signal "I'm ready", then waits until it can
connect to every other node's listener. Only after all nodes are confirmed
reachable does this script exit, allowing NCCL to connect on the first attempt.

Usage (in training script, BEFORE accelerate launch):
    python3 wait_for_all_nodes.py
"""

import os
import json
import socket
import time
import threading
import sys

# ---------------------------------------------------------------------------
# Parse cluster spec
# ---------------------------------------------------------------------------
spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
role = spec["role"]
workers = spec[role]
my_index = int(spec["index"])
nnodes = len(workers)

if nnodes <= 1:
    print("[Barrier] Single node, no barrier needed.")
    sys.exit(0)

# Parse all worker addresses (addr:port1,port2,...)
all_addrs = []
for w in workers:
    addr, ports = w.split(":")
    port = int(ports.split(",")[0])
    all_addrs.append((addr, port))

# Use a dedicated barrier port (offset from YARN-assigned port to avoid conflict)
BARRIER_OFFSET = 42
my_addr, my_port = all_addrs[my_index]
barrier_port = my_port + BARRIER_OFFSET

# ---------------------------------------------------------------------------
# Step 1: Start TCP listener to signal "I'm ready"
# ---------------------------------------------------------------------------
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", barrier_port))
server.listen(nnodes)

stop_event = threading.Event()

def accept_loop():
    while not stop_event.is_set():
        try:
            server.settimeout(1.0)
            conn, _ = server.accept()
            conn.close()
        except socket.timeout:
            continue
        except OSError:
            break

t = threading.Thread(target=accept_loop, daemon=True)
t.start()

print(f"[Barrier] Node {my_index}/{nnodes} ready, listening on :{barrier_port}")

# ---------------------------------------------------------------------------
# Step 2: Wait for every node to be reachable
# ---------------------------------------------------------------------------
TIMEOUT = 1800  # 30 min — if YARN can't schedule all nodes in 30 min, something is wrong
start = time.time()

for i in range(nnodes):
    addr, port = all_addrs[i]
    bp = port + BARRIER_OFFSET
    attempt = 0
    while True:
        elapsed = time.time() - start
        if elapsed > TIMEOUT:
            print(f"[Barrier] TIMEOUT after {elapsed:.0f}s waiting for node {i} ({addr}:{bp})")
            sys.exit(1)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((addr, bp))
            s.close()
            print(f"[Barrier] Node {i} ({addr}:{bp}) is alive (attempt {attempt})")
            break
        except (ConnectionRefusedError, socket.timeout, OSError):
            attempt += 1
            if attempt % 15 == 0:
                print(f"[Barrier] Still waiting for node {i} ({addr}:{bp}), "
                      f"attempt {attempt}, {elapsed:.0f}s elapsed...")
            time.sleep(2)

elapsed = time.time() - start
print(f"[Barrier] All {nnodes} nodes reachable in {elapsed:.1f}s. "
      f"Sleeping 5s for stabilization...")
time.sleep(5)
print(f"[Barrier] Launching training.")

# ---------------------------------------------------------------------------
# Step 3: Clean up
# ---------------------------------------------------------------------------
stop_event.set()
try:
    server.close()
except OSError:
    pass
t.join(timeout=2)
