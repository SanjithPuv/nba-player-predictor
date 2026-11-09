#!/bin/sh
set -eu

version=2.0.0
destination=$1

case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)
        platform=osx-arm64
        checksum=c35d60080b31ba5d11a3ce505a6831be784f003b4c575d32a4c2badb3b41cdf0
        ;;
    Darwin-x86_64)
        platform=osx-amd64
        checksum=98fef9fcad856555ca0403fe4d9d6680acfd26c67e8c6f004d61a248d6bc0eff
        ;;
    Linux-x86_64)
        platform=linux-amd64
        checksum=993ad89e7dfca66cd6da994c2cad090c2fc160ddf9f550498a098ab4bb13cfb1
        ;;
    Linux-aarch64|Linux-arm64)
        platform=linux-arm64
        checksum=c60eecf91143ca6bbac55c4f4d40dc22520488b0ac257c86e0fe9a2b451cf2b7
        ;;
    *)
        echo "Download DuckDB $version for your platform and set DUCKDB_DIR." >&2
        exit 1
        ;;
esac

archive=$(mktemp)
trap 'rm -f "$archive"' EXIT HUP INT TERM
asset="duckdb-shared-libs-$platform.tar.gz"
curl --fail --location --silent --show-error --retry 2 \
    "https://github.com/duckdb/duckdb/releases/download/v$version/$asset" -o "$archive"

if command -v sha256sum >/dev/null 2>&1; then
    printf '%s  %s\n' "$checksum" "$archive" | sha256sum -c -
else
    printf '%s  %s\n' "$checksum" "$archive" | shasum -a 256 -c -
fi

mkdir -p "$destination"
tar -xzf "$archive" -C "$destination"
