import psutil
import time
import os
import subprocess

def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:3.2f} {unit}"
        size /= 1024.0

def get_pstree_output(search_term="LM Studio"):
    """Run pstree and return the rendered process tree."""
    try:
        # Find the main process PID.
        pid_cmd = ["pgrep", "-x", search_term]
        main_pid = subprocess.check_output(pid_cmd).decode().strip().split('\n')[0]

        # Render the process tree.
        tree_cmd = ["pstree", "-p", main_pid]
        return subprocess.check_output(tree_cmd).decode().strip()
    except:
        return f"No main process named '{search_term}' was found (or pstree is not installed)."

def monitor_lm_studio_suite():
    target_keywords = ["LM Studio", "node"]
    
    try:
        while True:
            display_list = []
            # Capture the current process tree.
            tree_text = get_pstree_output("LM Studio")

            # Collect detailed memory metrics.
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
                try:
                    name = proc.info['name'] or ""
                    cmdline = " ".join(proc.info['cmdline'] or [])
                    
                    is_target = False
                    if "LM Studio" in name:
                        is_target = True
                    elif name.lower() == "node" and "LM Studio" in cmdline:
                        is_target = True
                        name = "Node (LM Studio Backend)"

                    if is_target:
                        mem = proc.info['memory_info']
                        display_list.append({
                            "name": name, "pid": proc.info['pid'],
                            "rss": format_bytes(mem.rss), "vms": format_bytes(mem.vms),
                            "rss_raw": mem.rss
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Render the dashboard.
            os.system('clear')
            print("="*30 + " LM Studio Process Tree " + "="*30)
            print(tree_text)
            print("\n" + "="*30 + " Live Memory Stats " + "="*30)
            print(f"{'Process Name':<30} | {'PID':<8} | {'Physical Memory (RSS)':<20} | {'Virtual Memory (VMS)':<20}")
            print("-" * 80)

            if not display_list:
                print("Waiting for LM Studio...")
            else:
                for p in sorted(display_list, key=lambda x: x['rss_raw'], reverse=True):
                    print(f"{p['name'][:30]:<30} | {p['pid']:<8} | {p['rss']:<15} | {p['vms']:<15}")

            print(f"\n[Last updated: {time.strftime('%H:%M:%S')}] (Ctrl+C to exit)")
            time.sleep(2)  # Lower the refresh rate to avoid calling pstree too frequently.

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    # Check whether pstree is installed.
    try:
        subprocess.run(["pstree", "-V"], capture_output=True)
    except FileNotFoundError:
        print("Error: 'pstree' was not found. Install it first with: brew install pstree")
    else:
        monitor_lm_studio_suite()
