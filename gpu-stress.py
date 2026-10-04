#!/usr/bin/env python3
"""Sustained per-GPU load test.

Loads an independent copy of a small model on each GPU (one worker process per
card) and keeps a fixed number of streaming requests in flight on each for a set
duration, while the parent process shows a live dashboard of temperature / power /
utilization (from nvidia-smi) and request / token throughput.

    .venv/bin/python gpu-stress.py                 # all GPUs, 10 minutes
    .venv/bin/python gpu-stress.py --duration 60   # one hour
    .venv/bin/python gpu-stress.py --gpus 0,2      # subset of cards
    .venv/bin/python gpu-stress.py --no-tui        # plain log lines instead of the dashboard
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import queue
import random
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Configuration
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"  # ~8GB of weights, fits a 16GB card
# Fraction of the card vLLM may claim (weights + KV cache). Leave headroom: the
# Mamba layers in Nemotron need ~1GB of scratch memory on top of this.
GPU_MEMORY_UTIL = 0.75
MAX_MODEL_LEN = 4096

CONCURRENCY = 64  # requests kept in flight per GPU
MAX_TOKENS = 512

PROMPTS = [
    "Explain quantum computing in detail.",
    "Write a short story about AI.",
    "What are the key differences between Python and Rust?",
    "Describe how a CPU executes an instruction, step by step.",
    "Write an essay on the history of the printing press.",
    "Explain how TCP congestion control works.",
    "Summarize the causes and consequences of the Industrial Revolution.",
    "Describe the life cycle of a star from nebula to remnant.",
]

GPU_QUERY = (
    "index,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total,"
    "fan.speed,pcie.link.gen.current,pcie.link.width.current,"
    "clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,"
    "clocks_event_reasons.hw_power_brake_slowdown"
)

# PCI subsystem vendor ids of common graphics card makers
CARD_VENDORS = {
    "0x1462": "MSI", "0x1458": "Gigabyte", "0x1043": "ASUS", "0x10de": "NVIDIA", "0x19da": "Zotac",
    "0x196e": "PNY", "0x3842": "EVGA", "0x1569": "Palit", "0x10b0": "Gainward", "0x1b4c": "Galax",
    "0x7377": "Colorful", "0x1f0a": "Inno3D",
}

SLOT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pci-slots.json")

RATE_WINDOW_S = 5  # window for the "tok/s now" figure
STATUS_STYLE = {
    "waiting": "dim",
    "loading": "yellow",
    "running": "bold green",
    "completed": "cyan",
    "failed": "bold red",
}


def get_gpu_stats():
    """Get current stats for every GPU, keyed by GPU index"""
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )

    def num(value):
        try:
            return float(value)
        except ValueError:  # "[N/A]" etc.
            return None

    stats = {}
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        stats[int(parts[0])] = {
            "temperature_c": num(parts[1]),
            "power_draw_w": num(parts[2]),
            "power_limit_w": num(parts[3]),
            "gpu_utilization_pct": num(parts[4]),
            "memory_used_mb": num(parts[5]),
            "memory_total_mb": num(parts[6]),
            "fan_speed_pct": num(parts[7]),
            "pcie_gen": num(parts[8]),
            "pcie_width": num(parts[9]),
            "throttled": any(p == "Active" for p in parts[10:13]),
        }
    return stats


def get_gpu_topology():
    """Where each GPU is plugged in: PCI address, card maker, and the slot it sits in"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Motherboard slot names (e.g. PCIE3) keyed by the bus address of the slot's port.
    # dmidecode needs root, so the result is cached for runs without passwordless sudo.
    slot_names = {}
    try:
        dmi = subprocess.run(["sudo", "-n", "dmidecode", "-t", "slot"], capture_output=True, text=True)
        designation = None
        for line in dmi.stdout.splitlines():
            key, _, value = line.strip().partition(": ")
            if key == "Designation":
                designation = value
            elif key == "Bus Address" and designation:
                slot_names[value.lower()] = designation
    except OSError:
        pass
    try:
        if slot_names:
            with open(SLOT_CACHE, "w") as f:
                json.dump(slot_names, f, indent=2)
        else:
            with open(SLOT_CACHE) as f:
                slot_names = json.load(f)
    except OSError:
        pass

    def read(path):
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return None

    topology = {}
    for line in result.stdout.strip().split("\n"):
        index, bus_id = [p.strip() for p in line.split(",")]
        address = bus_id[-12:].lower()  # nvidia-smi pads the domain to 8 digits
        device = f"/sys/bus/pci/devices/{address}"
        vendor_id = read(f"{device}/subsystem_vendor")
        # The parent device is the port (slot) the card is plugged into
        port = os.path.dirname(os.path.realpath(device))
        slot_width = read(f"{port}/max_link_width")
        topology[int(index)] = {
            "pci_address": address[5:],
            "card_vendor": CARD_VENDORS.get(vendor_id, vendor_id or "?"),
            "slot_name": slot_names.get(os.path.basename(port)),
            "slot_width": int(slot_width) if slot_width else None,
        }
    return topology


async def run_load(gpu, args, results, stop):
    """Load the model, then keep args.concurrency streaming requests in flight until stopped"""
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    load_start = time.time()
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_util,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=args.trust_remote_code,
            disable_log_stats=True,
        )
    )
    results.put({"gpu": gpu, "event": "loaded", "load_time_s": time.time() - load_start})

    # ignore_eos keeps every request generating for the full max_tokens, so the
    # load stays constant; DELTA streams tokens back so they can be counted live
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )
    counters = {"active": 0, "completed": 0, "input_tokens": 0, "output_tokens": 0}
    rng = random.Random(gpu)

    def next_prompt(slot, n):
        if not args.prompt_tokens:
            return PROMPTS[(slot + n) % len(PROMPTS)]
        # Long prompts make the load prefill-heavy, which draws more power than
        # decoding. Random token ids so no two prompts share a cacheable prefix.
        return {"prompt_token_ids": [rng.randrange(1000, 30000) for _ in range(args.prompt_tokens)]}

    async def client(slot):
        n = 0
        while True:
            counters["active"] += 1
            try:
                async for output in engine.generate(
                    next_prompt(slot, n), sampling_params, request_id=f"{slot}-{n}"
                ):
                    counters["output_tokens"] += len(output.outputs[0].token_ids)
                counters["input_tokens"] += len(output.prompt_token_ids or [])
                counters["completed"] += 1
            finally:
                counters["active"] -= 1
            n += 1

    clients = [asyncio.create_task(client(i)) for i in range(args.concurrency)]
    try:
        while not stop.is_set():
            await asyncio.sleep(0.5)
            results.put({"gpu": gpu, "event": "stats", **counters})
            for task in clients:
                if task.done():  # clients loop forever, so this is a failure
                    task.result()
    finally:
        for task in clients:
            task.cancel()
        await asyncio.gather(*clients, return_exceptions=True)
        engine.shutdown()


def worker(gpu, args, results, stop):
    """Runs in its own process, pinned to one GPU"""
    # Must be set before vllm/torch are imported so this process only sees its card
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # The FlashInfer sampler JIT-compiles a CUDA extension on first use (needs ninja
    # + nvcc on PATH); use the built-in torch sampler so the venv doesn't need activating
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
    log = open(f"{args.log_prefix}_gpu{gpu}.log", "w")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)

    try:
        asyncio.run(run_load(gpu, args, results, stop))
        results.put({"gpu": gpu, "event": "done"})
    except KeyboardInterrupt:  # Ctrl-C reaches the whole process group; the parent handles it
        results.put({"gpu": gpu, "event": "done"})
    except BaseException as e:
        import traceback

        traceback.print_exc()
        results.put({"gpu": gpu, "event": "error", "error": f"{type(e).__name__}: {e}"})
        raise


def describe_load(args):
    prompt = f"{args.prompt_tokens:,}-token prompt" if args.prompt_tokens else "short prompt"
    return f"{args.concurrency} concurrent requests per GPU, {prompt} -> {args.max_tokens} output tokens"


def fmt_duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def bar(fraction, width=8):
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def temp_style(temp):
    if temp is None:
        return "dim"
    return "green" if temp < 70 else "yellow" if temp < 80 else "bold red"


def util_style(util, running):
    if not running or util is None:
        return "dim"
    return "green" if util >= 90 else "yellow" if util >= 50 else "bold red"


def render(run):
    """Build the dashboard from the current run state"""
    args, gpus, state, stats = run["args"], run["gpus"], run["state"], run["stats"]
    now = time.time()

    # Header
    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold", justify="right")
    header.add_column()
    header.add_row("Model", args.model)
    header.add_row("Load", describe_load(args))
    if run["load_start"] is None:
        loaded = sum(1 for g in gpus if state[g]["status"] not in ("waiting", "loading"))
        header.add_row("Phase", Text(f"Loading models ({loaded}/{len(gpus)} ready)", style="yellow"))
    else:
        elapsed = now - run["load_start"]
        total = args.duration * 60
        progress = Text()
        progress.append(bar(elapsed / total, 30), style="green")
        progress.append(f"  {fmt_duration(elapsed)} / {fmt_duration(total)}")
        progress.append(f"   {fmt_duration(total - elapsed)} left", style="dim")
        header.add_row("Phase", Text("Under load", style="bold green"))
        header.add_row("Progress", progress)

    # Cards: where each GPU is plugged in, plus the slow-moving readings
    cards = Table(box=box.SIMPLE_HEAD, padding=(0, 1), title="Cards", title_justify="left", title_style="bold")
    for name, justify in (("GPU", "left"), ("Card", "left"), ("Bus", "left"), ("Slot", "left"),
                          ("Link", "left"), ("VRAM", "right"), ("Fan", "right")):
        cards.add_column(name, justify=justify, no_wrap=True, style="bold" if name == "GPU" else None)

    # Hardware: the live thermals and power
    hw = Table(box=box.SIMPLE_HEAD, padding=(0, 1), title="Hardware", title_justify="left", title_style="bold")
    for name, justify in (("GPU", "left"), ("Card", "left"), ("Status", "left"), ("Temp", "right"),
                          ("Peak", "right"), ("Power", "right"), ("", "left"), ("Util", "right"), ("", "left")):
        hw.add_column(name, justify=justify, no_wrap=True, style="bold" if name == "GPU" else None)

    total_power = 0.0
    for g in gpus:
        s, st, topo = state[g], stats[g], run["topology"][g]
        running = s["status"] == "running"
        temp, power, limit = st["temperature_c"], st["power_draw_w"] or 0, st["power_limit_w"]
        util = st["gpu_utilization_pct"]
        total_power += power
        status = Text(s["status"], style=STATUS_STYLE[s["status"]])
        if st["throttled"]:  # thermal or power-brake clock slowdown reported by the driver
            status = Text("THROTTLED", style="bold red")
        power_frac = power / limit if limit else 0
        power_style = "bold red" if power_frac >= 0.98 else "yellow" if power_frac >= 0.85 else "green"
        slot = " ".join(filter(None, [topo["slot_name"], f"x{topo['slot_width']}" if topo["slot_width"] else None]))
        cards.add_row(
            str(g),
            topo["card_vendor"],
            topo["pci_address"],
            slot or "?",
            f"Gen{st['pcie_gen']:.0f} x{st['pcie_width']:.0f}" if st["pcie_gen"] else "n/a",
            f"{(st['memory_used_mb'] or 0) / 1024:.1f}/{(st['memory_total_mb'] or 0) / 1024:.1f}G",
            f"{st['fan_speed_pct']:.0f}%" if st["fan_speed_pct"] is not None else "n/a",
        )
        hw.add_row(
            str(g),
            Text(f"{topo['card_vendor']} {slot}".strip(), style="dim"),
            status,
            Text(f"{temp:.0f}°C" if temp is not None else "n/a", style=temp_style(temp)),
            Text(f"{s['peak_temp_c']:.0f}°C" if s["peak_temp_c"] else "-", style=temp_style(s["peak_temp_c"])),
            f"{power:.0f}/{limit:.0f}W" if limit else f"{power:.0f}W",
            Text(bar(power_frac), style=power_style),
            Text(f"{util:.0f}%" if util is not None else "n/a", style=util_style(util, running)),
            Text(bar((util or 0) / 100), style=util_style(util, running)),
        )
    hw.add_section()
    hw.add_row("All", "", "", "", "", Text(f"{total_power:.0f}W", style="bold"))

    # Workload
    wl = Table(box=box.SIMPLE_HEAD, padding=(0, 1), title="Workload", title_justify="left", title_style="bold")
    wl.add_column("GPU", style="bold")
    wl.add_column("Running", justify="right")
    wl.add_column("Done", justify="right")
    wl.add_column("In tokens", justify="right")
    wl.add_column("Out tokens", justify="right")
    wl.add_column("Out tok/s", justify="right")
    wl.add_column("Avg", justify="right")
    wl.add_column("Load time", justify="right")

    totals = {"active": 0, "completed": 0, "input_tokens": 0, "output_tokens": 0, "rate_now": 0.0, "rate_avg": 0.0}
    for g in gpus:
        s = state[g]
        rate_now, rate_avg = token_rate(s, now), average_rate(s, now)
        for key, value in (("active", s["active"]), ("completed", s["completed"]), ("input_tokens", s["input_tokens"]),
                           ("output_tokens", s["output_tokens"]), ("rate_now", rate_now), ("rate_avg", rate_avg)):
            totals[key] += value
        active_style = "bold red" if s["status"] == "running" and s["active"] < args.concurrency else ""
        wl.add_row(
            str(g),
            Text(str(s["active"]), style=active_style),
            f"{s['completed']:,}",
            f"{s['input_tokens']:,}",
            f"{s['output_tokens']:,}",
            Text(f"{rate_now:,.0f}", style="bold cyan"),
            f"{rate_avg:,.0f}",
            f"{s['load_time_s']:.0f}s" if s["load_time_s"] else "-",
        )
    wl.add_section()
    wl.add_row(
        "All",
        str(totals["active"]),
        f"{totals['completed']:,}",
        f"{totals['input_tokens']:,}",
        f"{totals['output_tokens']:,}",
        Text(f"{totals['rate_now']:,.0f}", style="bold cyan"),
        f"{totals['rate_avg']:,.0f}",
        style="bold",
    )

    events = Text("\n".join(run["events"]) or "-", style="dim")
    footer = Text(f"Worker logs: {args.log_prefix}_gpu<N>.log   |   Ctrl-C to stop early", style="dim")
    return Group(
        Panel(header, title="GPU Stress Test", title_align="left", border_style="cyan"),
        cards,
        hw,
        wl,
        Panel(events, title="Events", title_align="left", border_style="dim"),
        footer,
    )


def token_rate(s, now):
    """Output tokens/sec over the last RATE_WINDOW_S seconds"""
    history = s["history"]
    if len(history) < 2 or now - history[-1][0] > RATE_WINDOW_S:
        return 0.0
    (t0, n0), (t1, n1) = history[0], history[-1]
    return (n1 - n0) / (t1 - t0) if t1 > t0 else 0.0


def average_rate(s, now):
    """Output tokens/sec since this GPU started generating"""
    if not s["started_at"]:
        return 0.0
    elapsed = (s["finished_at"] or now) - s["started_at"]
    return s["output_tokens"] / elapsed if elapsed > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Sustained per-GPU load test using vLLM")
    parser.add_argument("--duration", type=float, default=10, help="minutes under load (default 10)")
    parser.add_argument("--gpus", help="comma-separated GPU indices (default: all)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY, help="requests in flight per GPU")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--prompt-tokens", type=int, default=0,
                        help="use random prompts of this many tokens (prefill-heavy load, draws more power)")
    parser.add_argument("--gpu-memory-util", type=float, default=GPU_MEMORY_UTIL, help="fraction of VRAM vLLM may claim")
    parser.add_argument("--interval", type=float, default=10, help="seconds between recorded samples")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-tui", action="store_true", help="print plain status lines")
    args = parser.parse_args()
    if args.prompt_tokens + args.max_tokens > MAX_MODEL_LEN:
        parser.error(f"--prompt-tokens + --max-tokens must be <= {MAX_MODEL_LEN}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_prefix = f"stress_{run_id}"
    console = Console()
    use_tui = console.is_terminal and not args.no_tui

    stats = get_gpu_stats()
    idle_stats = stats
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else sorted(stats)

    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    stop = ctx.Event()
    procs = {g: ctx.Process(target=worker, args=(g, args, results, stop)) for g in gpus}

    state = {
        g: {"status": "waiting", "error": None, "load_time_s": None, "started_at": None,
            "finished_at": None, "active": 0, "completed": 0, "input_tokens": 0,
            "output_tokens": 0, "peak_temp_c": None, "history": deque()}
        for g in gpus
    }
    topology = get_gpu_topology()
    run = {"args": args, "gpus": gpus, "state": state, "stats": stats, "topology": topology, "load_start": None,
           "events": deque(maxlen=6)}
    samples = []

    def event(message):
        run["events"].append(f"{datetime.now().strftime('%H:%M:%S')}  {message}")
        if not use_tui:
            print(message, flush=True)

    def finish(g, status, error=None):
        s = state[g]
        s["status"], s["error"], s["active"] = status, error, 0
        s["finished_at"] = time.time()
        if error:
            event(f"GPU {g}: FAILED - {error}")

    def drain():
        while True:
            try:
                msg = results.get_nowait()
            except queue.Empty:
                return
            g, s = msg["gpu"], state[msg["gpu"]]
            if msg["event"] == "loaded":
                s["status"] = "running"
                s["load_time_s"] = round(msg["load_time_s"], 2)
                s["started_at"] = time.time()
                event(f"GPU {g}: model loaded in {msg['load_time_s']:.1f}s, generating")
            elif msg["event"] == "stats":
                for key in ("active", "completed", "input_tokens", "output_tokens"):
                    s[key] = msg[key]
                now = time.time()
                s["history"].append((now, msg["output_tokens"]))
                while now - s["history"][0][0] > RATE_WINDOW_S:
                    s["history"].popleft()
            elif msg["event"] == "done" and s["status"] in ("loading", "running"):
                finish(g, "completed" if s["started_at"] else "failed")
            elif msg["event"] == "error":
                finish(g, "failed", msg["error"])

    def check_dead():
        """A worker that dies without reporting (segfault, OOM kill, GPU fell off the bus)"""
        for g, p in procs.items():
            if p.pid is None:  # not started yet
                continue
            if state[g]["status"] in ("loading", "running") and not p.is_alive():
                drain()
                if state[g]["status"] in ("loading", "running"):
                    finish(g, "failed", f"worker exited unexpectedly (exit code {p.exitcode})")

    def with_status(*statuses):
        return [g for g in gpus if state[g]["status"] in statuses]

    if not use_tui:
        print("=" * 80)
        print(f"GPU Stress Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print(f"Model:    {args.model}")
        print(f"GPUs:     {gpus}")
        print(f"Duration: {args.duration:g} min under load")
        print(f"Load:     {describe_load(args)}")
        print(f"Logs:     {args.log_prefix}_gpu<N>.log")
        print("=" * 80)

    interrupted = False
    deadline = None
    last_stats = last_sample = 0.0
    live = Live(render(run), console=console, screen=True, auto_refresh=False) if use_tui else None
    try:
        if live:
            live.start()
        while True:
            drain()
            check_dead()
            now = time.time()

            # Load one card at a time; concurrent loads fight over disk and host RAM
            if run["load_start"] is None and not with_status("loading"):
                if with_status("waiting"):
                    g = with_status("waiting")[0]
                    state[g]["status"] = "loading"
                    procs[g].start()
                    event(f"GPU {g}: loading model")
                elif with_status("running"):
                    run["load_start"] = now
                    deadline = now + args.duration * 60
                    last_sample = now
                    event(f"All workers up, under load for {args.duration:g} min")
                else:
                    event("No workers started successfully")
                    break
            if deadline and (now >= deadline or not with_status("running")):
                break

            if now - last_stats >= 1:
                last_stats = now
                stats = run["stats"] = get_gpu_stats()
                for g in with_status("running"):
                    temp = stats[g]["temperature_c"]
                    if temp is not None:
                        state[g]["peak_temp_c"] = max(state[g]["peak_temp_c"] or 0, temp)

            if deadline and now - last_sample >= args.interval:
                last_sample = now
                elapsed = now - run["load_start"]
                samples.append(
                    {
                        "elapsed_s": round(elapsed, 1),
                        "gpus": {
                            g: {
                                **{k: v for k, v in stats[g].items() if not k.endswith(("_limit_w", "_total_mb"))},
                                "requests_running": state[g]["active"],
                                "output_tokens_per_second": round(token_rate(state[g], now), 1),
                            }
                            for g in gpus
                        },
                    }
                )
                if not use_tui:
                    print(f"[{elapsed / 60:6.1f} min]")
                    for g in gpus:
                        s, st = state[g], stats[g]
                        print(
                            f"  GPU {g}: {st['temperature_c']:.0f}°C  {st['power_draw_w']:6.1f}W  "
                            f"{st['gpu_utilization_pct']:3.0f}% util  {st['memory_used_mb']:6,.0f}MB  | "
                            f"{s['active']:3d} running  {s['completed']:6,d} done  "
                            f"{token_rate(s, now):7.0f} tok/s  {s['status']}",
                            flush=True,
                        )

            if live:
                live.update(render(run), refresh=True)
            time.sleep(0.25)
    except KeyboardInterrupt:
        interrupted = True
        event("Interrupted, stopping workers")
    finally:
        end_time = time.time()
        stop.set()
        if live:
            live.update(render(run), refresh=True)
        for p in procs.values():
            if p.is_alive():
                p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join()
        drain()
        if live:
            live.stop()

    load_time = end_time - run["load_start"] if run["load_start"] else 0
    for g in with_status("waiting", "loading", "running"):
        finish(g, "completed" if state[g]["output_tokens"] else "failed")

    # Compile results
    summary = {}
    for g in gpus:
        s = state[g]
        series = [x["gpus"][g] for x in samples]

        def agg(key, fn):
            values = [x[key] for x in series if x[key] is not None]
            return round(fn(values), 1) if values else None

        def mean(values):
            return sum(values) / len(values)

        summary[g] = {
            "status": s["status"],
            "error": s["error"],
            "model_load_time_s": s["load_time_s"],
            "requests_completed": s["completed"],
            "input_tokens": s["input_tokens"],
            "output_tokens": s["output_tokens"],
            "output_tokens_per_second": round(average_rate(s, end_time), 2),
            **topology[g],
            "idle": idle_stats[g],
            "temperature_c": {"max": agg("temperature_c", max), "avg": agg("temperature_c", mean)},
            "power_draw_w": {"max": agg("power_draw_w", max), "avg": agg("power_draw_w", mean)},
            "gpu_utilization_pct": {
                "min": agg("gpu_utilization_pct", min),
                "avg": agg("gpu_utilization_pct", mean),
            },
            "memory_used_mb": {"max": agg("memory_used_mb", max)},
            "fan_speed_pct": {"max": agg("fan_speed_pct", max)},
            "throttled": any(x["throttled"] for x in series),
        }

    results_json = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "configuration": {
            "gpus": gpus,
            "requested_duration_min": args.duration,
            "actual_load_time_s": round(load_time, 1),
            "interrupted": interrupted,
            "gpu_memory_utilization": args.gpu_memory_util,
            "max_model_len": MAX_MODEL_LEN,
            "concurrency_per_gpu": args.concurrency,
            "max_tokens_per_request": args.max_tokens,
            "prompt_tokens_per_request": args.prompt_tokens or None,
        },
        "summary": summary,
        "samples": samples,
    }

    print("\n" + "=" * 80)
    print("STRESS TEST RESULTS")
    print("=" * 80)
    print(f"Time under load: {load_time / 60:.1f} min" + (" (interrupted)" if interrupted else ""))
    for g in gpus:
        r = summary[g]
        print(f"\nGPU {g}: {r['status'].upper()}" + (f" - {r['error']}" if r["error"] else ""))
        print(f"  Requests:     {r['requests_completed']:,} completed")
        print(f"  Tokens:       {r['input_tokens']:,} input, {r['output_tokens']:,} output ({r['output_tokens_per_second']:.1f} tok/s)")
        print(f"  Temperature:  max {r['temperature_c']['max']}°C, avg {r['temperature_c']['avg']}°C (idle {idle_stats[g]['temperature_c']:.0f}°C)")
        print(f"  Power Draw:   max {r['power_draw_w']['max']}W, avg {r['power_draw_w']['avg']}W")
        print(f"  GPU Util:     avg {r['gpu_utilization_pct']['avg']}%, min {r['gpu_utilization_pct']['min']}%")
        print(f"  Memory Used:  max {r['memory_used_mb']['max']}MB")
        if r["throttled"]:
            print("  WARNING: clock throttling (thermal or power) was reported during the run")

    output_filename = f"stress_{run_id}.json"
    with open(output_filename, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_filename}")
    print("=" * 80)

    failed = [g for g in gpus if summary[g]["status"] != "completed"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
