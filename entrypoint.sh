#!/bin/bash

# Reload udev rules and trigger device detection (optional)
udevadm control --reload-rules && udevadm trigger

# Dynamically set permissions for USB devices
for dev in /dev/bus/usb/*/*; do
    if [ -e "$dev" ]; then
        chmod 0666 "$dev"
    fi
done

# Start the container's main process
exec "$@"
