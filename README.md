# ai-server-1

GPU stress test and results for a three-GPU inference server.

`gpu-stress.py` loads an independent copy of a small LLM onto each GPU with
[vLLM](https://github.com/vllm-project/vllm) (one worker process per card) and keeps a
fixed number of streaming requests in flight on every card for a set duration. A live
terminal dashboard shows temperature, power, utilization, PCIe link and token throughput
per card, and a JSON file with the full time series is written at the end.

## System under test

| Component | Details |
|---|---|
| Motherboard | MSI TRX40 PRO 10G (MS-7C60) |
| CPU | AMD Ryzen Threadripper 3960X, 24 cores / 48 threads |
| RAM | 64 GB |
| GPUs | 3 x NVIDIA GeForce RTX 5060 Ti 16 GB, 180 W power limit each |
| OS | Ubuntu 24.04.3 LTS, kernel 6.8.0 |
| NVIDIA driver | 580.173.02 (CUDA 13.0) |
| Software | Python 3.12, vLLM 0.30.0, PyTorch 2.13.0, Transformers 5.18.0 |
| Model | [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) (about 8 GB of weights) |

GPU placement:

| GPU | Card | PCI address | Motherboard slot | Link under load |
|---|---|---|---|---|
| 0 | MSI | `21:00.0` | PCIE7 (x16) | Gen4 x8 |
| 1 | MSI | `4a:00.0` | PCIE1 (x8) | Gen4 x8 |
| 2 | Gigabyte | `4b:00.0` | PCIE3 (x16) | Gen4 x8 |

The RTX 5060 Ti is an x8 card, so x8 is its full link width in every slot.

## Results (2026-10-04)

### Full power: 10 minutes, all three cards

`./gpu-stress.py --duration 10 --prompt-tokens 3000 --max-tokens 32`

Long random prompts make the load prefill-heavy, which holds each card at its power limit.

| GPU | Slot | Result | Power avg / max | Util avg | Temp idle / avg / max | Fan max | Throttled |
|---|---|---|---|---|---|---|---|
| 0 | PCIE7 | completed | 180.0 W / 180.4 W | 100% | 43 / 79.8 / 81 °C | 85% | no |
| 1 | PCIE1 | completed | 180.0 W / 180.2 W | 100% | 40 / 75.3 / 76 °C | 58% | no |
| 2 | PCIE3 | completed | 180.0 W / 180.2 W | 100% | 40 / 75.2 / 77 °C | 77% | no |

All three cards held their 180 W limit at 100% utilization for the full 10 minutes with
no worker failures and no thermal or power throttling reported by the driver. Every card
kept a Gen4 x8 link throughout. GPU 0 ran about 4-5 °C hotter than the other two. Across
the run the cards completed 3,698 requests (about 11.1 million input tokens).

Raw data: [`results/full-power_10min_20261004.json`](results/full-power_10min_20261004.json)

### Decode-heavy: 2 minutes, all three cards

`./gpu-stress.py --duration 2`

Short prompts with 512 output tokens each. This is closer to a normal generation
workload and draws less power.

| GPU | Result | Output tok/s | Power avg / max | Util avg | Temp avg / max |
|---|---|---|---|---|---|
| 0 | completed | 956 | 143.7 W / 145.1 W | 100% | 76.5 / 78 °C |
| 1 | completed | 967 | 149.0 W / 149.4 W | 100% | 67.2 / 68 °C |
| 2 | completed | 962 | 146.3 W / 148.9 W | 100% | 66.8 / 70 °C |

Raw data: [`results/decode_2min_20261004.json`](results/decode_2min_20261004.json)

Notes on reading the numbers:

- Cards load one at a time and start generating as soon as they are ready, while the
  timer starts once all cards are up. Request and token totals for earlier cards
  therefore include extra time and should not be compared across cards.
- Power, temperature and utilization statistics come from samples taken every 10
  seconds during the timed window only.

## Running it

Requirements: Linux, NVIDIA GPUs with a recent driver, Python 3.10+ and enough disk for
the model download. The pinned vLLM version needs a driver that supports CUDA 13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./gpu-stress.py                                              # all GPUs, 10 minutes
./gpu-stress.py --duration 60                                # one hour
./gpu-stress.py --gpus 0,2                                   # a subset of cards
./gpu-stress.py --prompt-tokens 3000 --max-tokens 32         # maximum power draw
./gpu-stress.py --no-tui                                     # plain log lines
```

| Option | Default | Meaning |
|---|---|---|
| `--duration` | 10 | Minutes under load, counted from when all cards are up |
| `--gpus` | all | Comma-separated GPU indices |
| `--model` | Nemotron 3 Nano 4B | Any vLLM-supported model that fits on one card |
| `--concurrency` | 64 | Requests kept in flight per GPU |
| `--max-tokens` | 512 | Output tokens per request |
| `--prompt-tokens` | 0 | Use random prompts of this many tokens (prefill-heavy) |
| `--gpu-memory-util` | 0.75 | Fraction of VRAM vLLM may claim |
| `--interval` | 10 | Seconds between recorded samples |

The script exits with status 1 if any card's worker fails or dies, so it can be used in
a burn-in loop.

Output files, written to the current directory:

- `stress_<timestamp>.json`: configuration, per-GPU summary and the sampled time series.
- `stress_<timestamp>_gpu<N>.log`: vLLM output for each worker. Look here first if a
  card fails.

### Slot names

The "Slot" column needs the motherboard's slot table, which only root can read. The
script tries `sudo -n dmidecode -t slot` and caches the result in `pci-slots.json`. The
file in this repository is for the MSI TRX40 PRO 10G above; delete it on other hardware.
Without it the dashboard still shows the PCI address and slot width.

## License

MIT
