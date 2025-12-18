#!/bin/bash
# Network tuning script for local multi-validator testing
# Works on macOS and Linux - sets optimal network parameters for current session

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Detect OS
OS="$(uname -s)"
info "Detected OS: $OS"

# Function to safely set sysctl value
set_sysctl() {
    local key="$1"
    local value="$2"
    local current

    if current=$(sysctl -n "$key" 2>/dev/null); then
        if [ "$current" -lt "$value" ] 2>/dev/null; then
            if sudo sysctl -w "${key}=${value}" >/dev/null 2>&1; then
                info "Set $key: $current -> $value"
            else
                warn "Failed to set $key (may require SIP disabled on macOS)"
            fi
        else
            info "$key already >= $value (current: $current)"
        fi
    else
        warn "Parameter $key not available on this system"
    fi
}

# Increase file descriptor limits
increase_fd_limits() {
    info "Increasing file descriptor limits..."

    local current_limit=$(ulimit -n)
    local target_limit=65536

    if [ "$current_limit" -lt "$target_limit" ]; then
        ulimit -n "$target_limit" 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true
        info "File descriptors: $current_limit -> $(ulimit -n)"
    else
        info "File descriptors already at $current_limit"
    fi
}

# macOS-specific tuning
tune_macos() {
    info "Applying macOS network tuning..."

    # TCP buffer sizes
    set_sysctl "net.inet.tcp.sendspace" 262144
    set_sysctl "net.inet.tcp.recvspace" 262144

    # UDP buffer size
    set_sysctl "net.inet.udp.maxdgram" 65535

    # Max socket buffer
    set_sysctl "kern.ipc.maxsockbuf" 8388608

    # Max pending connections
    set_sysctl "kern.ipc.somaxconn" 2048

    # Max sockets
    set_sysctl "kern.ipc.maxsockets" 32768

    # TCP connection limits
    set_sysctl "net.inet.tcp.msl" 15000

    # Disable delayed ACK for localhost (faster small messages)
    set_sysctl "net.inet.tcp.delayed_ack" 0

    # Increase ephemeral port range
    set_sysctl "net.inet.ip.portrange.first" 10000
    set_sysctl "net.inet.ip.portrange.last" 65535
}

# Linux-specific tuning
tune_linux() {
    info "Applying Linux network tuning..."

    # TCP buffer sizes (min, default, max)
    set_sysctl "net.core.rmem_max" 16777216
    set_sysctl "net.core.wmem_max" 16777216
    set_sysctl "net.core.rmem_default" 262144
    set_sysctl "net.core.wmem_default" 262144

    # TCP memory (in pages)
    set_sysctl "net.ipv4.tcp_rmem" "4096 262144 16777216"
    set_sysctl "net.ipv4.tcp_wmem" "4096 262144 16777216"

    # Max pending connections
    set_sysctl "net.core.somaxconn" 2048
    set_sysctl "net.ipv4.tcp_max_syn_backlog" 4096

    # Max sockets
    set_sysctl "net.core.netdev_max_backlog" 4096

    # Reduce TIME_WAIT
    set_sysctl "net.ipv4.tcp_fin_timeout" 15
    set_sysctl "net.ipv4.tcp_tw_reuse" 1

    # Disable delayed ACK for faster localhost
    set_sysctl "net.ipv4.tcp_low_latency" 1

    # Increase local port range
    set_sysctl "net.ipv4.ip_local_port_range" "10000 65535"

    # Max open files system-wide
    set_sysctl "fs.file-max" 2097152
    set_sysctl "fs.nr_open" 2097152
}

# Main
main() {
    echo ""
    echo "========================================"
    echo "  Network Tuning for Local Validators  "
    echo "========================================"
    echo ""

    increase_fd_limits

    case "$OS" in
        Darwin)
            tune_macos
            ;;
        Linux)
            tune_linux
            ;;
        *)
            error "Unsupported OS: $OS"
            exit 1
            ;;
    esac

    echo ""
    info "Network tuning complete!"
    echo ""
    echo "Current settings:"
    echo "  File descriptors: $(ulimit -n)"

    if [ "$OS" = "Darwin" ]; then
        echo "  TCP send buffer:  $(sysctl -n net.inet.tcp.sendspace 2>/dev/null || echo 'N/A')"
        echo "  TCP recv buffer:  $(sysctl -n net.inet.tcp.recvspace 2>/dev/null || echo 'N/A')"
        echo "  Max connections:  $(sysctl -n kern.ipc.somaxconn 2>/dev/null || echo 'N/A')"
    else
        echo "  TCP rmem max:     $(sysctl -n net.core.rmem_max 2>/dev/null || echo 'N/A')"
        echo "  TCP wmem max:     $(sysctl -n net.core.wmem_max 2>/dev/null || echo 'N/A')"
        echo "  Max connections:  $(sysctl -n net.core.somaxconn 2>/dev/null || echo 'N/A')"
    fi
    echo ""
    warn "Note: These settings are temporary and will reset on reboot."
    echo ""
}

main "$@"
