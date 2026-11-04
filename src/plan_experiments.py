"""Write stage-2 configs; does not execute training or generation."""
from copy import deepcopy
from pathlib import Path
from .common import read_json, write_json


def main():
    base = read_json("configs/baseline.json")
    configs = []
    control = deepcopy(base)
    control.update(experiment="control_only", method="control_only")
    configs.append(control)
    for scale in [0.3, 0.6, 0.9]:
        c = deepcopy(base)
        c.update(experiment=f"ip_const_{scale}", adapter_scale=scale)
        configs.append(c)
    for rank in [4, 16]:
        for scale in [0.5, 0.8, 1.1]:
            c = deepcopy(base)
            c.update(experiment=f"lora_r{rank}_const_{scale}", method="lora", adapter_scale=scale,
                     lora_path=f"outputs/lora_r{rank}")
            configs.append(c)
    for method, weight in [("ip_adapter", 0.6), ("lora", 0.8)]:
        for schedule in ["linear_up", "linear_down"]:
            c = deepcopy(base)
            prefix = "ip" if method == "ip_adapter" else "lora_r4"
            c.update(experiment=f"{prefix}_{schedule}_{weight}", method=method,
                     adapter_scale=weight, schedule=schedule)
            if method == "lora":
                c["lora_path"] = "outputs/lora_r4"
            configs.append(c)
    for c in configs:
        write_json(Path("configs/experiments") / (c["experiment"] + ".json"), c)
    print(f"Prepared {len(configs)} dev configurations. Run on dev (5 photos), not the final test set.")


if __name__ == "__main__":
    main()
