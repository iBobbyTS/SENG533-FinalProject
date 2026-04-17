import psutil
import subprocess
import re
import os

def parse_size_to_bytes(size_str):
    """Convert vmmap sizes such as 19.8G or 100K into bytes."""
    if not size_str:
        return 0
    size_str = size_str.upper().strip()
    multiplier = 1
    if size_str.endswith('K'): multiplier = 1024
    elif size_str.endswith('M'): multiplier = 1024**2
    elif size_str.endswith('G'): multiplier = 1024**3
    
    try:
        val = float(re.sub(r'[A-Z]', '', size_str))
        return int(val * multiplier)
    except ValueError:
        return 0

def format_bytes(size):
    """Format bytes into a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:6.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def find_target_pid():
    """Find the LM Studio-related node process with the largest RSS."""
    max_rss = 0
    target_pid = None
    target_name = ""

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            name = proc.info['name'] or ""
            cmdline = " ".join(proc.info['cmdline'] or [])
            # Match node processes whose command line contains LM Studio.
            if "node" in name.lower() and "lm studio" in cmdline.lower():
                rss = proc.info['memory_info'].rss
                if rss > max_rss:
                    max_rss = rss
                    target_pid = proc.info['pid']
                    target_name = name
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
            
    return target_pid, target_name

def analyze_vmmap(pid):
    print(f"Analyzing PID: {pid} (this may take a few seconds)...\n")
    try:
        output = subprocess.check_output(['vmmap', '-summary', str(pid)], stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        print("vmmap failed to run. Make sure you have permission, or try running with sudo.")
        return

    # Extract the global physical footprint.
    footprint_match = re.search(r'Physical footprint:\s+([0-9\.]+[KMG])', output)
    overall_footprint = footprint_match.group(1) if footprint_match else "Unknown"

    # Initialize category accumulators in bytes.
    categories = {
        "1. Model (static weights)": {"VIRTUAL": 0, "RESIDENT": 0, "SWAPPED": 0},
        "2. Context & GPU (context and GPU buffers)": {"VIRTUAL": 0, "RESIDENT": 0, "SWAPPED": 0},
        "3. Software (framework, UI, stacks, libraries)": {"VIRTUAL": 0, "RESIDENT": 0, "SWAPPED": 0}
    }

    # Match vmmap summary rows and extract region name, virtual memory,
    # resident memory, dirty memory (ignored), and swapped memory.
    row_pattern = re.compile(r'^([A-Za-z_][A-Za-z0-9_\s\.\(\)]+?)\s+([0-9\.]+[KMG])\s+([0-9\.]+[KMG])\s+([0-9\.]+[KMG])\s+([0-9\.]+[KMG])', re.MULTILINE)

    for match in row_pattern.finditer(output):
        region_name = match.group(1).strip()
        virt = parse_size_to_bytes(match.group(2))
        res = parse_size_to_bytes(match.group(3))
        swap = parse_size_to_bytes(match.group(5))

        # Core classification logic.
        if region_name.startswith("mapped file"):
            # Model weights are usually file-mapped.
            cat = "1. Model (static weights)"
        elif region_name.startswith("VM_ALLOCATE") or region_name.startswith("shared memory") or region_name.startswith("IOAccelerator"):
            # Preallocated KV cache and shared Metal/GPU compute memory.
            cat = "2. Context & GPU (context and GPU buffers)"
        elif region_name.startswith("TOTAL"):
            continue
        else:
            # Remaining MALLOC, __TEXT, __DATA, stacks, and similar regions
            # are treated as software/runtime overhead.
            cat = "3. Software (framework, UI, stacks, libraries)"

        categories[cat]["VIRTUAL"] += virt
        categories[cat]["RESIDENT"] += res
        categories[cat]["SWAPPED"] += swap

    # Render the formatted report.
    os.system('clear')
    print("="*75)
    print(f" LM Studio Memory Classification Report (PID: {pid})")
    print(f" Reported Physical Footprint: {overall_footprint}")
    print("="*75)
    print(f"{'Category':<48} | {'Resident Memory':<22} | {'Swapped'}")
    print("-" * 75)

    for cat_name, data in sorted(categories.items()):
        res_str = format_bytes(data['RESIDENT'])
        swap_str = format_bytes(data['SWAPPED'])
        print(f"{cat_name:<48} | {res_str:<22} | {swap_str}")

    print("-" * 75)
    print("Notes:")
    print(" - [Model] should be close to the model file size.")
    print(" - [Context] changes the most; changing Context Length mainly affects this bucket.")
    print(" - If [Context] or [Software] has swapped memory > 0, slowdown is likely occurring.")

if __name__ == "__main__":
    pid, name = find_target_pid()
    if pid:
        analyze_vmmap(pid)
    else:
        print("No running LM Studio backend node process was found. Load a model first.")
