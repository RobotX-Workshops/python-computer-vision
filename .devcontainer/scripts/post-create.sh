#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
echo "Repository root: $REPO_ROOT"
VENV_PATH="$REPO_ROOT/.venv"

sudo rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/partial/*
APT_BASE_CMD="sudo apt-get install -y --no-install-recommends libgl1 libglib2.0-0 cmake build-essential"

sudo apt-get update

if ! $APT_BASE_CMD; then
	sudo apt-get update --allow-releaseinfo-change
	if ! $APT_BASE_CMD; then
		printf 'Retrying apt install with --fix-missing due to archive issues...\n'
		$APT_BASE_CMD --fix-missing
	fi
fi
sudo apt-get clean

if [ ! -d "$VENV_PATH" ]; then
	python3 -m venv "$VENV_PATH"
fi

"$VENV_PATH/bin/python" -m pip install --upgrade pip

BASHRC_FILE="$HOME/.bashrc"
MARKER_START="# >>> python-computer-vision venv >>>"
MARKER_END="# <<< python-computer-vision venv <<<"

# Ensure every interactive shell sources the workshop virtual environment.
if [ ! -f "$BASHRC_FILE" ]; then
	touch "$BASHRC_FILE"
fi

if grep -Fq "$MARKER_START" "$BASHRC_FILE"; then
	sed -i "/$MARKER_START/,/$MARKER_END/d" "$BASHRC_FILE"
fi

{
	echo "$MARKER_START"
	echo ". \"$VENV_PATH/bin/activate\""
	echo "$MARKER_END"
} >> "$BASHRC_FILE"