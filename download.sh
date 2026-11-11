#!/usr/bin/env bash
set -e

REPO="prakash-2409/Deep-RL-lab"
BRANCH="main"
BASE_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}"

FILES=(
    "ex1.py"
    "ex2.py"
    "ex3.py"
    "ex3.m"
    "ex4.py"
    "ex5.py"
    "ex6.py"
    "ex7.py"
    "ex7.m"
    "ex8.py"
    "ex8.m"
    "ex9.py"
    "ex10.py"
    "ex11.py"
)

echo "Downloading Deep RL Lab experiments from ${REPO}..."
for file in "${FILES[@]}"; do
    echo "Fetching ${file}..."
    curl -sSL "${BASE_URL}/${file}" -o "${file}"
done

echo "All experiment files downloaded successfully!"
ls -la ex*.py ex*.m 2>/dev/null || true
