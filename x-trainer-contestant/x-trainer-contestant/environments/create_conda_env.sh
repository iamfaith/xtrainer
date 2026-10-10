#!/usr/bin/env bash
# Linux x86_64; run with bash, not source.
set -euo pipefail
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo 'Run with bash, not source.' >&2; return 2; fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
usage() {
    echo 'Usage: bash create_conda_env.sh'
    echo '   or: bash create_conda_env.sh YAML NEW_ENV_NAME [--dry-run]'
    echo 'Optional: MIRROR=ustc|original, CONDA_BASE=/path/to/conda'
    echo 'The Dobot-Sim environment installs this repository source/leisaac automatically.'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ $# == 0 ]]; then
    bash "$0" lerobot.yaml lerobot
    bash "$0" Dobot-Sim.yaml Dobot-Sim
    echo 'Both environments are ready.'
    exit 0
fi
[[ $# == 2 || ( $# == 3 && "$3" == --dry-run ) ]] || { usage >&2; exit 2; }
env_file="$1"; env_name="$2"; dry_run=0
[[ $# == 3 ]] && dry_run=1
[[ -f "$env_file" ]] || env_file="$script_dir/$1"
[[ -f "$env_file" ]] || { echo "YAML not found: $1" >&2; exit 2; }
env_file="$(cd -- "$(dirname -- "$env_file")" && pwd -P)/$(basename -- "$env_file")"
[[ "$env_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ && "$env_name" != base ]] || { echo 'Choose a non-base environment name.' >&2; exit 2; }
mirror="${MIRROR:-ustc}"
[[ "$mirror" == ustc || "$mirror" == original ]] || { echo 'MIRROR must be ustc or original.' >&2; exit 2; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'These exports require Linux x86_64.' >&2; exit 1; }
conda_base="${CONDA_BASE:-}"
if [[ -z "$conda_base" ]] && command -v conda >/dev/null 2>&1; then conda_base="$(conda info --base)"; fi
if [[ -z "$conda_base" ]]; then
    for candidate in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" /opt/conda; do
        if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then conda_base="$candidate"; break; fi
    done
fi
[[ -f "$conda_base/etc/profile.d/conda.sh" ]] || { echo 'Conda not found; set CONDA_BASE.' >&2; exit 1; }
source "$conda_base/etc/profile.d/conda.sh"
conda activate base
work_dir="$(mktemp -d)"
trap 'rm -f -- "$work_dir/conda.yaml" "$work_dir/pip.txt"; rmdir -- "$work_dir"' EXIT
python - "$env_file" "$env_name" "$mirror" "$work_dir" "$script_dir/.." <<'PY'
import json, os, re, sys
from pathlib import Path
try:
    from ruamel.yaml import YAML
    read_yaml = YAML(typ='safe').load
except ImportError:
    try:
        import yaml
        read_yaml = yaml.safe_load
    except ImportError:
        raise SystemExit('Conda base needs ruamel.yaml or PyYAML to read the export.')
src, name, mirror, tmp, project = sys.argv[1:]
data = read_yaml(Path(src).read_text(encoding='utf-8-sig'))
if not isinstance(data, dict) or not isinstance(data.get('dependencies'), list):
    raise SystemExit('Invalid Conda environment YAML')
conda_deps, pip_deps = [], []
for item in data['dependencies']:
    if isinstance(item, str): conda_deps.append(item)
    elif isinstance(item, dict) and set(item) == {'pip'}: pip_deps.extend(item['pip'])
    else: raise SystemExit(f'Unsupported dependency entry: {item!r}')
if not any(re.match(r'^pip(?:[=<>!]|$)', p) for p in conda_deps): conda_deps.append('pip')
channels = list(data.get('channels', []))
if mirror == 'ustc':
    channels = ['https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge'
                if c in ('conda-forge', 'https://conda.anaconda.org/conda-forge') else c for c in channels]
# Keep pkgs/main at its exported mirror: USTC does not directly mirror it.
if 'nodefaults' not in channels: channels.append('nodefaults')
index = 'https://mirrors.ustc.edu.cn/pypi/simple' if mirror == 'ustc' else 'https://pypi.org/simple'
options = [f'--index-url {index}']
cuda, install_deps, editable = set(), [], []
for req in pip_deps:
    if not isinstance(req, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+==[^\s]+', req):
        raise SystemExit(f'Expected a pinned package; restore local/URL requirement manually: {req!r}')
    pkg = req.split('==')[0].lower().replace('_', '-')
    if pkg in ('torch', 'torchvision', 'torchaudio'):
        match = re.search(r'\+(cu\d+)$', req)
        if match: cuda.add(match[1])
    if pkg == 'leisaac':
        p = (Path(project) / 'source/leisaac').resolve()
        if not any((p / f).is_file() for f in ('pyproject.toml', 'setup.py')):
            raise SystemExit(f'Installable source directory missing: {p}')
        editable.append(f'-e "{p}"')
    else:
        install_deps.append(req)
if len(cuda) > 1: raise SystemExit(f'Mixed CUDA wheel variants: {cuda}')
for variant in sorted(cuda): options.append(f'--extra-index-url https://download.pytorch.org/whl/{variant}')
if any(p.startswith('isaacsim==') for p in pip_deps): options.append('--extra-index-url https://pypi.nvidia.com')
Path(tmp, 'conda.yaml').write_text(json.dumps({'name': name, 'channels': channels, 'dependencies': conda_deps}, indent=2)+'\n')
Path(tmp, 'pip.txt').write_text('\n'.join(options+install_deps+editable)+'\n')
print(f'Source YAML: {src}\nTarget: {name}\nChannels: {channels}')
print('\n'.join(options + editable))
print(f'Pinned dependencies: {len(conda_deps)} Conda, {len(install_deps)} pip; CUDA: {sorted(cuda)}')
PY
if ((dry_run)); then
    echo 'Preflight passed. No environment created; package availability/resolution not tested.'
    exit 0
fi
# Existing environments are refused. Failed installs remain available for diagnosis.
conda env create --name "$env_name" --file "$work_dir/conda.yaml" --yes
conda activate "$env_name"
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_NO_INDEX PIP_FIND_LINKS
export PIP_CONFIG_FILE=/dev/null
python -m pip install --timeout 120 --retries 5 -r "$work_dir/pip.txt"
python -m pip check
python -c 'import torch; print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU available:", torch.cuda.is_available())'
printf 'Environment ready: %s\nActivate in your terminal: conda activate %s\n' "$env_name" "$env_name"
