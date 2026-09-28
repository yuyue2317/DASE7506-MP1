import subprocess
import psutil
import time

cmd = [
    "python",
    "evaluate.py",
    "--checkpoint", "runs/e03-rope-2400/checkpoint.pt",
    "--device", "cpu",
    "--precision", "fp32",
    "--split", "test",
]

process = subprocess.Popen(cmd)
parent = psutil.Process(process.pid)

peak_rss = 0

while process.poll() is None:
    try:
        processes = [parent] + parent.children(recursive=True)

        total_rss = 0
        for p in processes:
            try:
                total_rss += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        peak_rss = max(peak_rss, total_rss)

    except psutil.NoSuchProcess:
        pass

    time.sleep(0.01)

process.wait()

peak_gib = peak_rss / (1024 ** 3)

print()
print("=" * 50)
print(f"Peak RAM: {peak_gib:.3f} GiB")
print(f"Limit:    4.000 GiB")
print(f"PASS:     {peak_gib <= 4.0}")
print("=" * 50)