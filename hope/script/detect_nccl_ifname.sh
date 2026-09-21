#!/bin/bash
# Detect the correct network interface for NCCL TCP bootstrap.
#
# The old heuristic (first eth* interface) grabs the RDMA/RoCE NIC (e.g. eth9)
# which does NOT allow TCP connections — only RDMA verbs. This causes NCCL
# bootstrap to time out because it can't make TCP connections over that NIC.
#
# Instead, we ask the kernel which interface *routes* to a peer node.
# This always returns the correct TCP-capable interface.
#
# Usage:  source detect_nccl_ifname.sh
# Effect: sets NCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME, TP_SOCKET_IFNAME

IF_NAME=$(python3 -c "
import os, json, socket, subprocess
spec = json.loads(os.environ['AFO_ENV_CLUSTER_SPEC'])
role = spec['role']
my_idx = int(spec['index'])
workers = spec[role]
# Pick a peer node (not ourselves) to test routing
peer_idx = (my_idx + 1) % len(workers)
peer_host = workers[peer_idx].split(':')[0]
peer_ip = socket.gethostbyname(peer_host)
# Ask kernel which interface routes to that IP
out = subprocess.check_output(['ip', 'route', 'get', peer_ip], text=True)
tokens = out.split()
print(tokens[tokens.index('dev') + 1])
" 2>/dev/null)
IF_NAME=${IF_NAME:-eth0}
echo "Detected NCCL_SOCKET_IFNAME=${IF_NAME}  (route-based, avoids RDMA-only NIC)"
export NCCL_SOCKET_IFNAME=${IF_NAME}
export GLOO_SOCKET_IFNAME=${IF_NAME}
export TP_SOCKET_IFNAME=${IF_NAME}
