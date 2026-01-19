#!/bin/bash
set -e

export PKG_CONFIG_PATH="/usr/lib/aarch64-linux-gnu/pkgconfig"

exec "$@"
