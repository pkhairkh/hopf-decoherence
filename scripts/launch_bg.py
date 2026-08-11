#!/usr/bin/env python3
"""Launch a long-running background command on the remote and return immediately."""
import sys
import paramiko
import time

HOST = "192.248.158.130"
USER = "root"
PASS = ".6Pb}XGKr[xd8j.7"

def launch(cmd):
    """Launch a background command via setsid, return immediately."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30)
    # Wrap in setsid + nohup so it survives channel close
    full = f'setsid bash -c {repr(cmd)} < /dev/null > /dev/null 2>&1 & echo "PID=$!"'
    stdin, stdout, stderr = client.exec_command(full, timeout=10)
    out = stdout.read().decode('utf-8', errors='replace')
    # Close the channel without waiting
    client.close()
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: launch_bg.py '<command>'", file=sys.stderr)
        sys.exit(2)
    print(launch(" ".join(sys.argv[1:])))
