"""Hugging Face browsing, download, and cache-management helpers."""

from __future__ import annotations

import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download

from axolotl_lcars_ui.resources import format_bytes


RepoType = Literal["model", "dataset"]

MODEL_DOWNLOAD_ALLOW = (
    "*.json",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.model",
    "*.txt",
    "*.tiktoken",
    "*.spm",
    "*.py",
    "*.md",
)
DATASET_DOWNLOAD_ALLOW = (
    "*.json",
    "*.jsonl",
    "*.parquet",
    "*.csv",
    "*.arrow",
    "*.txt",
    "*.py",
    "*.md",
)
MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
RUNTIME_MODEL_SUFFIXES = (".gguf", ".onnx", ".engine", ".tflite", ".mlx")
DATASET_SUFFIXES = (".json", ".jsonl", ".parquet", ".csv", ".arrow", ".txt")


@dataclass
class SearchResult:
    repo_id: str
    repo_type: RepoType
    downloads: int | None = None
    likes: int | None = None
    tags: str = ""
    updated: str = ""
    size: str = ""
    pipeline: str = ""
    library: str = ""
    gated: str = ""
    sha: str = ""
    base_models: str = ""
    children: str = ""
    file_count: int = 0
    role: str = ""
    weights: str = ""
    quants: str = ""
    compatibility: str = ""
    blocked: bool = False


@dataclass
class RepoFile:
    path: str
    size: int = 0
    kind: str = ""
    axolotl: str = ""


@dataclass
class RepoDetails:
    result: SearchResult
    files: list[RepoFile] = field(default_factory=list)


@dataclass
class DownloadJob:
    job_id: str
    repo_id: str
    repo_type: RepoType
    revision: str | None = None
    status: str = "queued"
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    local_path: str = ""
    error: str = ""


class HuggingFaceManager:
    """Thin wrapper around the official Hugging Face Hub client."""

    def __init__(self) -> None:
        self.api = HfApi()
        self.search_results: list[SearchResult] = []
        self.related_results: list[SearchResult] = []
        self.selected_details: RepoDetails | None = None
        self.jobs: dict[str, DownloadJob] = {}
        self.logs: deque[str] = deque(maxlen=300)
        self.last_repo_id = ""
        self.last_repo_type: RepoType = "model"
        self.last_local_path = ""
        self._lock = threading.Lock()

    def search(
        self,
        query: str,
        repo_type: RepoType,
        *,
        limit: int = 12,
        sort: str = "downloads",
        compatible_only: bool = True,
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            self.log("Search skipped: empty query.")
            return self.search_results
        sort = _normalize_sort(sort)
        self.log(f"Searching Hugging Face {repo_type}s for {query!r} sorted by {sort}.")
        try:
            if repo_type == "model":
                items = self.api.list_models(
                    search=query,
                    sort=sort,
                    limit=limit,
                    full=True,
                    token=self._token(),
                )
            else:
                items = self.api.list_datasets(
                    search=query,
                    sort=sort,
                    limit=limit,
                    full=True,
                    token=self._token(),
                )
            results = [self._result_from_info(item, repo_type) for item in items]
            if compatible_only:
                results = [item for item in results if not item.blocked]
        except Exception as exc:  # network/auth/client errors are user-visible in the log
            if sort != "downloads":
                self.log(f"HF search failed with sort={sort}: {exc}; retrying downloads sort.")
                return self.search(query, repo_type, limit=limit, sort="downloads", compatible_only=compatible_only)
            self.log(f"HF search failed: {exc}")
            return self.search_results

        with self._lock:
            self.search_results = results
            self.related_results = []
            self.selected_details = None
            if results:
                self.last_repo_id = results[0].repo_id
                self.last_repo_type = repo_type
        self.log(f"Search returned {len(results)} {repo_type} result(s).")
        return results

    def inspect_repo(
        self,
        repo_id: str,
        repo_type: RepoType,
        *,
        revision: str | None = None,
    ) -> RepoDetails | None:
        repo_id = repo_id.strip()
        if not repo_id:
            self.log("Inspect skipped: repo id is empty.")
            return self.selected_details
        self.log(f"Inspecting {repo_type} {repo_id}.")
        try:
            if repo_type == "model":
                info = self.api.model_info(
                    repo_id,
                    revision=revision or None,
                    files_metadata=True,
                    token=self._token(),
                )
            else:
                info = self.api.dataset_info(
                    repo_id,
                    revision=revision or None,
                    files_metadata=True,
                    token=self._token(),
                )
        except Exception as exc:
            self.log(f"HF inspect failed for {repo_id}: {exc}")
            return self.selected_details
        result = self._result_from_info(info, repo_type)
        files = [_repo_file_from_sibling(item, repo_type) for item in _siblings(info)]
        if files:
            result.file_count = len(files)
            if not result.size:
                result.size = format_bytes(sum(item.size for item in files))
        details = RepoDetails(result=result, files=files)
        with self._lock:
            self.selected_details = details
            self.last_repo_id = repo_id
            self.last_repo_type = repo_type
        self.log(f"Inspect found {len(files)} file(s) for {repo_id}.")
        return details

    def select_result(self, repo_id: str) -> SearchResult | None:
        repo_id = repo_id.strip()
        for result in self.search_results:
            if result.repo_id == repo_id:
                with self._lock:
                    self.last_repo_id = result.repo_id
                    self.last_repo_type = result.repo_type
                return result
        return None

    def find_related_models(self, repo_id: str, *, limit: int = 12) -> list[SearchResult]:
        repo_id = repo_id.strip()
        if not repo_id:
            self.log("Fine-tune search skipped: repo id is empty.")
            return self.related_results
        self.log(f"Searching HF model lineage for fine-tunes of {repo_id}.")
        seen: set[str] = set()
        related: list[SearchResult] = []
        for fetch in (
            lambda: self.api.list_models(
                filter=f"base_model:{repo_id}",
                sort="downloads",
                limit=limit,
                full=True,
                token=self._token(),
            ),
            lambda: self.api.list_models(
                search=repo_id.split("/")[-1],
                sort="downloads",
                limit=limit * 2,
                full=True,
                token=self._token(),
            ),
        ):
            try:
                for item in fetch():
                    result = self._result_from_info(item, "model")
                    if result.repo_id == repo_id or result.repo_id in seen or result.blocked:
                        continue
                    if _is_related_to(result, repo_id):
                        seen.add(result.repo_id)
                        related.append(result)
                        if len(related) >= limit:
                            break
            except Exception as exc:
                self.log(f"HF fine-tune lookup partial failure: {exc}")
            if len(related) >= limit:
                break
        with self._lock:
            self.related_results = related
        self.log(f"Fine-tune lookup returned {len(related)} result(s).")
        return related

    def start_download(
        self,
        repo_id: str,
        repo_type: RepoType,
        *,
        revision: str | None = None,
    ) -> DownloadJob:
        repo_id = repo_id.strip()
        if not repo_id:
            raise ValueError("Repo ID is required.")
        job_id = f"{repo_type}:{repo_id}:{int(time.time())}"
        job = DownloadJob(job_id=job_id, repo_id=repo_id, repo_type=repo_type, revision=revision)
        with self._lock:
            self.jobs[job_id] = job
            self.last_repo_id = repo_id
            self.last_repo_type = repo_type
        thread = threading.Thread(target=self._download_worker, args=(job_id,), daemon=True)
        thread.start()
        self.log(f"Queued HF download: {repo_type} {repo_id}.")
        return job

    def cache_rows(self) -> tuple[list[dict[str, str]], str, int]:
        try:
            info = scan_cache_dir()
        except Exception as exc:
            self.log(f"HF cache scan failed: {exc}")
            return [], "0B", 0
        rows = []
        for repo in sorted(info.repos, key=lambda item: item.size_on_disk, reverse=True):
            revisions = sorted(repo.revisions, key=lambda rev: rev.last_modified, reverse=True)
            latest = revisions[0] if revisions else None
            rows.append(
                {
                    "Type": str(repo.repo_type),
                    "Repo": repo.repo_id,
                    "Size": repo.size_on_disk_str,
                    "Files": str(repo.nb_files),
                    "Revision": "" if latest is None else latest.commit_hash[:12],
                    "Path": str(repo.repo_path),
                }
            )
        return rows, info.size_on_disk_str, int(info.size_on_disk)

    def delete_repo(self, repo_id: str, repo_type: RepoType) -> str:
        repo_id = repo_id.strip()
        if not repo_id:
            raise ValueError("Repo ID is required.")
        info = scan_cache_dir()
        matches = [
            repo
            for repo in info.repos
            if repo.repo_id == repo_id and str(repo.repo_type or "model") == repo_type
        ]
        if not matches:
            raise ValueError(f"No cached {repo_type} repo found for {repo_id}.")
        revisions = [rev.commit_hash for repo in matches for rev in repo.revisions]
        strategy = info.delete_revisions(*revisions)
        expected = strategy.expected_freed_size_str
        strategy.execute()
        self.log(f"Deleted cached {repo_type} {repo_id}; expected freed space {expected}.")
        return expected

    def job_rows(self) -> list[dict[str, str]]:
        with self._lock:
            jobs = sorted(self.jobs.values(), key=lambda job: job.started_at, reverse=True)
        return [
            {
                "Repo": job.repo_id,
                "Type": job.repo_type,
                "Status": job.status,
                "Revision": job.revision or "main",
                "Local Path": job.local_path or job.error,
            }
            for job in jobs[:8]
        ]

    def drain_logs(self) -> list[str]:
        lines = list(self.logs)
        self.logs.clear()
        return lines

    def log(self, message: str) -> None:
        self.logs.append(f"[HF] {message}")

    def _download_worker(self, job_id: str) -> None:
        with self._lock:
            job = self.jobs[job_id]
            job.status = "running"
        try:
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            local_path = snapshot_download(
                repo_id=job.repo_id,
                repo_type=job.repo_type,
                revision=job.revision or None,
                allow_patterns=_allow_patterns(job.repo_type),
                token=token or None,
            )
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.ended_at = time.time()
            self.log(f"Download failed for {job.repo_type} {job.repo_id}: {exc}")
            return

        with self._lock:
            job.status = "complete"
            job.local_path = local_path
            job.ended_at = time.time()
            self.last_local_path = local_path
        self.log(f"Download complete for {job.repo_type} {job.repo_id}: {local_path}")

    def _result_from_info(self, item: object, repo_type: RepoType) -> SearchResult:
        repo_id = str(getattr(item, "id", "") or getattr(item, "repo_id", ""))
        tags = getattr(item, "tags", None) or []
        updated = getattr(item, "last_modified", None) or getattr(item, "lastModified", None)
        siblings = _siblings(item)
        filenames = [_sibling_path(sibling) for sibling in siblings]
        classification = _classify_repo(repo_type, repo_id, filenames, tags)
        base_models = getattr(item, "base_models", None) or getattr(item, "baseModels", None) or []
        if isinstance(base_models, str):
            base_models = [base_models]
        children_count = getattr(item, "children_model_count", None) or getattr(item, "childrenModelCount", None)
        size = _repo_size(item, siblings)
        return SearchResult(
            repo_id=repo_id,
            repo_type=repo_type,
            downloads=getattr(item, "downloads", None),
            likes=getattr(item, "likes", None),
            tags=", ".join(str(tag) for tag in tags[:5]),
            updated="" if updated is None else str(updated)[:19],
            size="" if size <= 0 else format_bytes(size),
            pipeline=str(getattr(item, "pipeline_tag", None) or ""),
            library=str(getattr(item, "library_name", None) or ""),
            gated=_flag(getattr(item, "gated", None)),
            sha=str(getattr(item, "sha", "") or "")[:12],
            base_models=", ".join(str(model) for model in base_models[:3]),
            children="" if children_count is None else f"{children_count:,}",
            file_count=len(filenames),
            role=classification["role"],
            weights=classification["weights"],
            quants=classification["quants"],
            compatibility=classification["compatibility"],
            blocked=classification["blocked"] == "true",
        )

    def _token(self) -> str | bool | None:
        return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None


def search_rows(results: list[SearchResult]) -> list[dict[str, str]]:
    if not results:
        return [{"Repo": "No results yet", "Role": "", "Axolotl": "", "Size": "", "Files": "", "Quants": "", "Downloads": "", "Likes": ""}]
    return [
        {
            "Repo": item.repo_id,
            "Role": item.role,
            "Axolotl": item.compatibility,
            "Size": item.size,
            "Files": str(item.file_count or ""),
            "Quants": item.quants,
            "Downloads": "" if item.downloads is None else f"{item.downloads:,}",
            "Likes": "" if item.likes is None else f"{item.likes:,}",
            "Updated": item.updated,
        }
        for item in results
    ]


def result_options(results: list[SearchResult]) -> list[str]:
    if not results:
        return [""]
    return [result.repo_id for result in results]


def detail_summary_rows(details: RepoDetails | None) -> list[dict[str, str]]:
    if details is None:
        return [{"Field": "Selected", "Value": "No repository inspected yet"}]
    result = details.result
    rows = [
        ("Repo ID", result.repo_id),
        ("Type", result.repo_type),
        ("Axolotl Role", result.role),
        ("Compatibility", result.compatibility),
        ("Size", result.size),
        ("Weights / Files", result.weights),
        ("Quants", result.quants),
        ("Pipeline", result.pipeline),
        ("Library", result.library),
        ("Base Models", result.base_models),
        ("Fine-tune Count", result.children),
        ("Gated", result.gated),
        ("Revision", result.sha),
        ("Tags", result.tags),
    ]
    return [{"Field": key, "Value": value} for key, value in rows if value]


def detail_file_rows(details: RepoDetails | None) -> list[dict[str, str]]:
    if details is None:
        return [{"File": "Inspect a repo to see compatible files", "Size": "", "Kind": "", "Axolotl": ""}]
    if not details.files:
        return [{"File": "No files exposed by Hub metadata", "Size": "", "Kind": "", "Axolotl": details.result.compatibility}]
    return [
        {
            "File": item.path,
            "Size": format_bytes(item.size) if item.size else "",
            "Kind": item.kind,
            "Axolotl": item.axolotl,
        }
        for item in sorted(details.files, key=lambda file: (file.axolotl.startswith("skip"), file.path.lower()))[:80]
    ]


def related_rows(results: list[SearchResult]) -> list[dict[str, str]]:
    if not results:
        return [{"Repo": "No related fine-tunes loaded", "Role": "", "Axolotl": "", "Size": "", "Downloads": ""}]
    return [
        {
            "Repo": item.repo_id,
            "Role": item.role,
            "Axolotl": item.compatibility,
            "Size": item.size,
            "Downloads": "" if item.downloads is None else f"{item.downloads:,}",
        }
        for item in results
    ]


def cache_summary_text(total_bytes: int, total_text: str) -> str:
    return total_text if total_text != "0B" else format_bytes(total_bytes)


def _normalize_sort(sort: str) -> str:
    value = (sort or "downloads").strip()
    return value if value in {"downloads", "likes", "last_modified", "trending_score"} else "downloads"


def _allow_patterns(repo_type: RepoType) -> tuple[str, ...]:
    return MODEL_DOWNLOAD_ALLOW if repo_type == "model" else DATASET_DOWNLOAD_ALLOW


def _siblings(item: object) -> list[object]:
    siblings = getattr(item, "siblings", None) or []
    return list(siblings) if isinstance(siblings, (list, tuple)) else []


def _sibling_path(item: object) -> str:
    return str(
        getattr(item, "rfilename", None)
        or getattr(item, "filename", None)
        or getattr(item, "path", None)
        or getattr(item, "name", None)
        or ""
    )


def _sibling_size(item: object) -> int:
    value = getattr(item, "size", None)
    if isinstance(value, (int, float)):
        return int(value)
    lfs = getattr(item, "lfs", None)
    if isinstance(lfs, dict):
        value = lfs.get("size")
    else:
        value = getattr(lfs, "size", None)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _repo_size(item: object, siblings: list[object]) -> int:
    for attr in ("used_storage", "usedStorage"):
        value = getattr(item, attr, None)
        if isinstance(value, (int, float)):
            return int(value)
    safetensors = getattr(item, "safetensors", None)
    total = getattr(safetensors, "total", None)
    if isinstance(total, (int, float)):
        return int(total)
    if isinstance(safetensors, dict):
        total = safetensors.get("total")
        if isinstance(total, (int, float)):
            return int(total)
    return sum(_sibling_size(sibling) for sibling in siblings)


def _classify_repo(repo_type: RepoType, repo_id: str, filenames: list[str], tags: list[object]) -> dict[str, str]:
    if repo_type == "dataset":
        return _classify_dataset(filenames)
    lowered_names = [name.lower() for name in filenames if name]
    lowered_tags = [str(tag).lower() for tag in tags]
    repo_lower = repo_id.lower()
    has_config = "config.json" in lowered_names
    has_tokenizer = any(name.endswith(("tokenizer.json", "tokenizer.model", "tokenizer_config.json")) for name in lowered_names)
    adapter_files = [name for name in lowered_names if name.endswith(("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"))]
    weight_files = [
        name
        for name in lowered_names
        if name.endswith(MODEL_WEIGHT_SUFFIXES)
        and not name.endswith(("adapter_model.safetensors", "adapter_model.bin"))
    ]
    runtime_files = [name for name in lowered_names if name.endswith(RUNTIME_MODEL_SUFFIXES)]
    quants = _quant_summary(repo_lower, lowered_names, lowered_tags)
    if adapter_files and not weight_files:
        return {
            "role": "peft_adapter",
            "weights": f"{len(adapter_files)} adapter file(s)",
            "quants": quants,
            "compatibility": "OK as lora_model_dir with a matching base_model",
            "blocked": "false",
        }
    if weight_files and has_config:
        token_note = "" if has_tokenizer else "; tokenizer may need tokenizer_config"
        return {
            "role": "base_model",
            "weights": f"{len(weight_files)} HF weight file(s)",
            "quants": quants,
            "compatibility": f"OK: Transformers weights{token_note}",
            "blocked": "false",
        }
    if runtime_files and not weight_files:
        runtime_kind = "GGUF/runtime quant" if any(name.endswith(".gguf") for name in runtime_files) else "runtime artifact"
        return {
            "role": "runtime_quant",
            "weights": f"{len(runtime_files)} runtime file(s)",
            "quants": quants or runtime_kind,
            "compatibility": "Blocked: not an Axolotl base_model",
            "blocked": "true",
        }
    if "gguf" in repo_lower and not weight_files:
        return {
            "role": "runtime_quant",
            "weights": "",
            "quants": quants or "GGUF",
            "compatibility": "Blocked: likely GGUF-only runtime repo",
            "blocked": "true",
        }
    if weight_files:
        return {
            "role": "base_model?",
            "weights": f"{len(weight_files)} HF weight file(s)",
            "quants": quants,
            "compatibility": "Warn: weights found, inspect config/tokenizer files",
            "blocked": "false",
        }
    if not filenames:
        return {
            "role": "unknown",
            "weights": "",
            "quants": quants,
            "compatibility": "Warn: inspect files before download",
            "blocked": "false",
        }
    return {
        "role": "unsupported",
        "weights": "",
        "quants": quants,
        "compatibility": "Blocked: no Axolotl-readable weights",
        "blocked": "true",
    }


def _classify_dataset(filenames: list[str]) -> dict[str, str]:
    lowered = [name.lower() for name in filenames if name]
    data_files = [name for name in lowered if name.endswith(DATASET_SUFFIXES)]
    if data_files:
        suffixes = sorted({_suffix(name) for name in data_files if _suffix(name)})
        return {
            "role": "dataset",
            "weights": f"{len(data_files)} compatible data file(s)",
            "quants": ", ".join(suffixes),
            "compatibility": "OK: datasets-compatible files",
            "blocked": "false",
        }
    if not filenames:
        return {
            "role": "dataset",
            "weights": "",
            "quants": "",
            "compatibility": "Warn: inspect files before download",
            "blocked": "false",
        }
    return {
        "role": "dataset",
        "weights": "",
        "quants": "",
        "compatibility": "Blocked: no json/jsonl/parquet/csv/arrow/text files",
        "blocked": "true",
    }


def _repo_file_from_sibling(item: object, repo_type: RepoType) -> RepoFile:
    path = _sibling_path(item)
    lower = path.lower()
    if repo_type == "dataset":
        if lower.endswith(DATASET_SUFFIXES):
            kind = _suffix(lower).lstrip(".") or "dataset"
            axolotl = "download/use"
        else:
            kind = "metadata" if lower.endswith((".md", ".py", ".json")) else "other"
            axolotl = "support" if kind == "metadata" else "skip"
        return RepoFile(path=path, size=_sibling_size(item), kind=kind, axolotl=axolotl)
    if lower.endswith(("config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", ".model", ".tiktoken")):
        kind = "config/tokenizer"
        axolotl = "download/use"
    elif lower.endswith(("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")):
        kind = "PEFT adapter"
        axolotl = "download/use as lora_model_dir"
    elif lower.endswith(MODEL_WEIGHT_SUFFIXES):
        kind = "HF weight"
        axolotl = "download/use"
    elif lower.endswith(RUNTIME_MODEL_SUFFIXES):
        kind = "runtime quant"
        axolotl = "skip"
    elif lower.endswith((".md", ".txt", ".py", ".json")):
        kind = "support"
        axolotl = "download/use"
    else:
        kind = "other"
        axolotl = "skip"
    return RepoFile(path=path, size=_sibling_size(item), kind=kind, axolotl=axolotl)


def _quant_summary(repo_id: str, filenames: list[str], tags: list[str]) -> str:
    text = " ".join([repo_id, *filenames[:80], *tags])
    found: list[str] = []
    for pattern, label in (
        (r"\bgguf\b|\.gguf\b", "GGUF"),
        (r"\bexl2\b", "EXL2"),
        (r"\bgptq\b", "GPTQ"),
        (r"\bawq\b", "AWQ"),
        (r"\bbnb\b|bitsandbytes", "bitsandbytes"),
        (r"\bfp8\b", "FP8"),
        (r"\bbf16\b|bfloat16", "BF16"),
        (r"\bfp16\b|float16", "FP16"),
        (r"\bint8\b|8bit|8-bit|q8[_-]", "8-bit"),
        (r"\bint4\b|4bit|4-bit|q4[_-]", "4-bit"),
        (r"\bq5[_-]", "5-bit"),
    ):
        if re.search(pattern, text):
            found.append(label)
    return ", ".join(dict.fromkeys(found))


def _is_related_to(result: SearchResult, repo_id: str) -> bool:
    needle = repo_id.lower()
    leaf = repo_id.split("/")[-1].lower()
    haystack = " ".join([result.base_models, result.tags, result.repo_id]).lower()
    return needle in haystack or leaf in haystack


def _flag(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _suffix(value: str) -> str:
    index = value.rfind(".")
    return "" if index < 0 else value[index:]
