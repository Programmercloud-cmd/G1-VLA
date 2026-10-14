"""Portable, standard-library configuration shared by the pipeline commands."""
import copy
import json
import math
import os
import shutil
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def executable_path(value):
    path = Path(value).expanduser()
    return path.absolute() if path.is_absolute() else (PROJECT_ROOT / path).absolute()


def load_config(path=None):
    selected = path or os.environ.get("VLA_CONFIG")
    config_path = resolve_path(selected) if selected else PROJECT_ROOT / "machine.local.json"
    supplied = json.loads(config_path.read_text()) if config_path.exists() else {}
    if selected and not config_path.exists():
        raise FileNotFoundError(config_path)
    allowed = {"paths", "interpreters", "tools", "storage", "reserve_gib", "python_versions"}
    if set(supplied) - allowed:
        raise ValueError("Unknown configuration sections: " + ", ".join(sorted(set(supplied)-allowed)))
    defaults = {"raw_data":"data/g1_inspire_pick_up_black_box_clean", "derived":"derived",
        "reports":"reports", "runs":"/dev/shm/vla-groot-smoke", "archive":"storage",
        "cache":".cache", "envs":".venvs", "vendor":"vendor", "tools":".tools"}
    overrides = supplied.get("paths", {})
    valid_paths = set(defaults) | {"hf_cache", "model_aliases", "runtime_record"}
    if set(overrides) - valid_paths:
        raise ValueError("Unknown configured path")
    paths = {k:resolve_path(overrides.get(k,v)) for k,v in defaults.items()}
    paths["hf_cache"] = resolve_path(overrides["hf_cache"]) if "hf_cache" in overrides else paths["cache"] / "huggingface/hub"
    paths["model_aliases"] = resolve_path(overrides["model_aliases"]) if "model_aliases" in overrides else paths["tools"] / "model-aliases"
    paths["runtime_record"] = resolve_path(overrides["runtime_record"]) if "runtime_record" in overrides else paths["reports"] / "machine-runtime.json"
    raw = paths["raw_data"]
    for key,value in paths.items():
        if key != "raw_data" and (value == raw or value.is_relative_to(raw)):
            raise ValueError("Writable path overlaps original recordings: " + key)
    sources = {"unitree":paths["vendor"] / "unitree_lerobot", "conversion":paths["vendor"] / "lerobot-conversion", "gr00t":paths["vendor"] / "Isaac-GR00T"}
    interpreters = {"unitree":paths["envs"] / "unitree/bin/python", "conversion":paths["envs"] / "conversion/bin/python", "gr00t":sources["gr00t"] / ".venv/bin/python"}
    if set(supplied.get("interpreters",{})) - set(interpreters):
        raise ValueError("Unknown interpreter role")
    interpreters.update({k:executable_path(v) for k,v in supplied.get("interpreters",{}).items()})
    tools = {"uv":paths["tools"] / "python/bin/uv", "python":paths["tools"] / "python/bin/python", "ffmpeg_bin":paths["tools"] / "ffmpeg/usr/bin", "ffmpeg_lib":paths["tools"] / "ffmpeg/usr/lib/x86_64-linux-gnu"}
    if set(supplied.get("tools",{})) - set(tools):
        raise ValueError("Unknown tool path")
    tools.update({k:executable_path(v) for k,v in supplied.get("tools",{}).items()})
    if "ffmpeg_bin" not in supplied.get("tools",{}) and not (tools["ffmpeg_bin"] / "ffmpeg").exists() and shutil.which("ffmpeg"):
        tools["ffmpeg_bin"] = Path(shutil.which("ffmpeg")).parent
        if "ffmpeg_lib" not in supplied.get("tools",{}):
            tools["ffmpeg_lib"] = Path("/usr/lib/x86_64-linux-gnu")
    for value in [*interpreters.values(), tools["python"], tools["uv"]]:
        if value.parent.resolve().is_relative_to(raw):
            raise ValueError("Executable installation path overlaps original recordings")
    storage = {"enabled":False, "host":None, "remote_path":None}
    if set(supplied.get("storage",{})) - set(storage):
        raise ValueError("Unknown storage setting")
    storage.update(supplied.get("storage",{}));storage["mount"] = paths["archive"]
    if storage["enabled"] and (not storage["host"] or not storage["remote_path"]):
        raise ValueError("Enabled SSH storage requires host and remote_path")
    reserve = float(supplied.get("reserve_gib",8))
    if not math.isfinite(reserve) or reserve < 8:
        raise ValueError("reserve_gib must be at least 8")
    return {"paths":paths,"sources":sources,"interpreters":interpreters,"tools":tools,"storage":storage,
            "reserve_gib":reserve,"python_versions":supplied.get("python_versions",{"conversion":"3.10.12","gr00t":"3.12.2"}),"config_path":config_path}


def require_storage(config):
    mount = config["paths"]["archive"]
    if not os.path.ismount(mount):
        raise ValueError("Configured external storage is not mounted: " + str(mount))
    if config["storage"]["enabled"]:
        result = subprocess.run(["findmnt","--json","--target",str(mount)],capture_output=True,text=True,check=True)
        filesystems = json.loads(result.stdout).get("filesystems",[])
        expected = config["storage"]["host"] + ":" + config["storage"]["remote_path"]
        if not any(x.get("target")==str(mount) and x.get("source")==expected and x.get("fstype")=="fuse.sshfs" for x in filesystems):
            raise ValueError("External mount source differs from configured SSHFS source")
    return mount


def guard_path(path, config, kind):
    value = resolve_path(path)
    raw = config["paths"]["raw_data"].resolve()
    if value == raw or value.is_relative_to(raw):
        raise ValueError("Original recordings are read-only")
    if kind not in config["paths"] or kind == "raw_data":
        raise ValueError("Unknown writable path kind")
    base = config["paths"][kind].resolve()
    archive = config["paths"]["archive"].resolve()
    allowed = [base]
    if kind == "derived":
        allowed.append(archive / "derived")
    if not any(value.is_relative_to(root) for root in allowed):
        raise ValueError("Artifact is outside configured " + kind + " directory")
    if value.is_relative_to(archive) and config["storage"]["enabled"]:
        require_storage(config)
    return value


def guard_derived(path, config):
    return guard_path(path,config,"derived")


def load_runtime(config):
    models = json.loads((PROJECT_ROOT / "locks/model-sources.json").read_text())
    records = {repo:{"revision":rec["revision"],"snapshot":config["paths"]["hf_cache"] / ("models--"+repo.replace("/","--")) / "snapshots" / rec["revision"]} for repo,rec in models.items()}
    return {"sources":copy.copy(config["sources"]),"interpreters":copy.copy(config["interpreters"]),
            "tools":copy.copy(config["tools"]),"models":records,"source_lock":PROJECT_ROOT / "locks/runtime-sources.json"}


def runtime_environment(config, offline=True):
    env = os.environ.copy()
    for key in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN","HUGGINGFACE_TOKEN","HUGGINGFACEHUB_API_TOKEN","HUGGINGFACE_API_TOKEN"):
        env.pop(key,None)
    env.update(HF_HUB_CACHE=str(config["paths"]["hf_cache"]),HF_HUB_OFFLINE="1" if offline else "0",
        TRANSFORMERS_OFFLINE="1" if offline else "0",HF_HUB_DISABLE_XET="1",WANDB_DISABLED="true",
        NO_ALBUMENTATIONS_UPDATE="1",PYTHONDONTWRITEBYTECODE="1",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="4",
        UV_CACHE_DIR=str(config["paths"]["cache"] / "uv"),UV_LINK_MODE="hardlink",
        UV_PYTHON_INSTALL_DIR=str(config["paths"]["tools"] / "pythons"),VLA_CONFIG=str(config["config_path"]))
    if not config["config_path"].exists():
        env.pop("VLA_CONFIG",None)
    env["PATH"] = str(config["tools"]["ffmpeg_bin"]) + ":" + env.get("PATH","")
    env["LD_LIBRARY_PATH"] = str(config["tools"]["ffmpeg_lib"]) + (":"+env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env
