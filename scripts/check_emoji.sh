#!/bin/bash

set -e

BASE_COMMIT=$1
HEAD_COMMIT=$2

if [ -z "$BASE_COMMIT" ] || [ -z "$HEAD_COMMIT" ]; then
  echo "Usage: $0 <base_commit> <head_commit>"
  exit 1
fi

echo "Checking for emojis in lines added by the PR..."

# Scan only added lines, so emojis already on the base branch do not fail the PR
git diff -U0 "$BASE_COMMIT...$HEAD_COMMIT" | python3 -c "
import re
import sys

emoji_pattern = re.compile(
    '[\U0001F300-\U0001F9FF'  # Miscellaneous Symbols and Pictographs, Emoticons, etc.
    '\U00002600-\U000026FF'   # Miscellaneous Symbols
    '\U00002700-\U000027BF'   # Dingbats
    '\U0001F600-\U0001F64F'   # Emoticons
    '\U0001F680-\U0001F6FF'   # Transport and Map
    '\U0001F900-\U0001F9FF]'  # Supplemental Symbols
)

# Track the file and new-side line number from each diff hunk header
found = False
file = None
line_num = 0
for raw in sys.stdin.buffer:
    line = raw.decode('utf-8', errors='replace').rstrip('\n')
    if line.startswith('+++ '):
        file = line[6:] if line.startswith('+++ b/') else None
    elif line.startswith('@@'):
        line_num = int(re.search(r'\+(\d+)', line).group(1))
    elif line.startswith('+') and file is not None:
        if emoji_pattern.search(line):
            print(f'ERROR: Emoji in {file}:{line_num}: {line[1:].strip()}')
            found = True
        line_num += 1

if found:
    print('Please remove all emojis from code, comments, and documentation.')
    sys.exit(1)
print('PASS: No emojis found in PR changes')
"
