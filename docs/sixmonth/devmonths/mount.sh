#!/bin/bash
# VM: overlay the OS-disk repository on the local NVMe (writes go to /mnt/scratch/up; wiped at deallocation).
set -e
R=/home/azureuser/prc-flight-2026
if ! mountpoint -q /mnt/scratch; then sudo mkfs.ext4 -q -F /dev/nvme2n1; sudo mkdir -p /mnt/scratch; sudo mount /dev/nvme2n1 /mnt/scratch; sudo chown azureuser:azureuser /mnt/scratch; fi
mkdir -p /mnt/scratch/up /mnt/scratch/wk /mnt/scratch/trace
mount | grep -q " $R type overlay" || sudo mount -t overlay overlay -o lowerdir=$R,upperdir=/mnt/scratch/up,workdir=/mnt/scratch/wk $R
mount | grep " $R "; df -h /mnt/scratch | tail -1
