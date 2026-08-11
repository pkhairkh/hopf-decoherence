#!/usr/bin/env python3
"""SSH command executor for remote AMD EPYC-Turin sandbox."""
import sys
import paramiko
import os

HOST = "192.248.158.130"
USER = "root"
PASS = ".6Pb}XGKr[xd8j.7"

def run(cmd, timeout=600):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30,
                   banner_timeout=30, auth_timeout=30)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    rc = stdout.channel.recv_exit_status()
    client.close()
    return rc, out, err

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ssh_exec.py '<command>' [timeout]", file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    rc, out, err = run(cmd, timeout=timeout)
    sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    sys.exit(rc)
