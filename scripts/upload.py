#!/usr/bin/env python3
"""Upload a local file to the remote sandbox."""
import sys
import paramiko

HOST = "192.248.158.130"
USER = "root"
PASS = ".6Pb}XGKr[xd8j.7"

def upload(local_path, remote_path):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()
    client.close()
    print(f"Uploaded {local_path} -> {remote_path}")

if __name__ == "__main__":
    upload(sys.argv[1], sys.argv[2])
