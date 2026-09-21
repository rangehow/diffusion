import os
import json

cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
role = cluster_spec["role"]
assert role == "worker", "{} vs worker".format(role)
node_rank = cluster_spec["index"]
nnodes = len(cluster_spec[role])
def get_device_count():
    """
    获取当前节点的设备数量，兼容 Ascend NPU 和 NVIDIA GPU
    优先级：
    1. Ascend 910B
    2. Ascend 910C / Ascend910
    3. NVIDIA GPU
    """
    # 先尝试 Ascend NPU
    try:
        import torch_npu  # 如果能 import，说明是 NPU 环境
        # Ascend 910B
        count = os.popen("npu-smi info | grep 910B | wc -l").read().strip()
        if count.isdigit() and int(count) > 0:
            return int(count)

        # Ascend 910C
        count = os.popen("npu-smi info | grep -E '910B|Ascend910' | wc -l").read().strip()
        if count.isdigit() and int(count) > 0:
            return int(count)
    except ImportError:
        # torch_npu 不存在，说明大概率是 NVIDIA
        pass


    # 再尝试 NVIDIA
    try:
        count = os.popen("nvidia-smi --list-gpus | wc -l").read().strip()
        if count.isdigit() and int(count) > 0:
            return int(count)
    except Exception as e:
        ...

    # 如果都没有检测到，返回 0
    return 0

nproc_per_node = get_device_count()
total_processes = nnodes * nproc_per_node

master = cluster_spec[role][0]
master_addr, master_ports = master.split(":")
master_ports = master_ports.split(",")

print(
    "accelerate launch "
    "--multi_gpu "
    "--mixed_precision bf16 "
    "--num_processes={} "
    "--num_machines={} "
    "--machine_rank={} "
    "--main_process_ip={} "
    "--main_process_port={}".format(
        total_processes, # <--- 使用计算出的总进程数
        nnodes,
        node_rank,
        master_addr,
        master_ports[0]
    )
)