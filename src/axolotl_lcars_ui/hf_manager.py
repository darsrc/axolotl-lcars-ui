"""Hugging Face browsing, download, and cache-management helpers."""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

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
BYTES_PER_GIB = 1024**3
DOWNLOAD_SPACE_BUFFER = 1.1
MODEL_SUPPORT_DOWNLOAD_ALLOW = tuple(
    pattern for pattern in MODEL_DOWNLOAD_ALLOW if pattern not in {"*.safetensors", "*.bin", "*.pt"}
)
DATASET_SUPPORT_DOWNLOAD_ALLOW = ("*.py", "*.md")


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
    size_bytes: int = 0
    weight_bytes: int = 0
    params: str = ""
    role: str = ""
    weights: str = ""
    quants: str = ""
    fit: str = ""
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
    allow_patterns: tuple[str, ...] = field(default_factory=tuple)
    estimated_bytes: int = 0


class HuggingFaceManager:
    """Thin wrapper around the official Hugging Face Hub client."""

    def __init__(self) -> None:
        self.api = HfApi()
        self.all_search_results: list[SearchResult] = []
        self.search_results: list[SearchResult] = []
        self.related_results: list[SearchResult] = []
        self.selected_details: RepoDetails | None = None
        self.jobs: dict[str, DownloadJob] = {}
        self.logs: deque[str] = deque(maxlen=300)
        self.last_repo_id = ""
        self.last_repo_type: RepoType = "model"
        self.last_local_path = ""
        self.vram_limit_gb: float | None = None
        self.max_concurrent_downloads = _download_concurrency()
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
                items = self._list_models(query=query, sort=sort, limit=limit)
            else:
                items = self._list_datasets(query=query, sort=sort, limit=limit)
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
            self.all_search_results = results
            self.search_results = results
            self.related_results = []
            self.selected_details = None
            if results:
                self.last_repo_id = results[0].repo_id
                self.last_repo_type = repo_type
        self.log(f"Search returned {len(results)} {repo_type} result(s).")
        return results

    def sift_results(
        self,
        *,
        text: str = "",
        sort: str = "downloads",
        artifact_filter: str = "any",
        quant_filter: str = "any",
        fit_filter: str = "any",
        vram_limit_gb: float | None = None,
    ) -> list[SearchResult]:
        text = text.strip().lower()
        sort = _normalize_local_sort(sort)
        artifact_filter = artifact_filter or "any"
        quant_filter = quant_filter or "any"
        fit_filter = fit_filter or "any"
        self.vram_limit_gb = vram_limit_gb if vram_limit_gb and vram_limit_gb > 0 else None
        results = []
        for item in self.all_search_results:
            result = _with_fit(item, self.vram_limit_gb)
            haystack = " ".join(
                [
                    result.repo_id,
                    result.tags,
                    result.pipeline,
                    result.library,
                    result.weights,
                    result.quants,
                    result.compatibility,
                ]
            ).lower()
            if text and text not in haystack:
                continue
            if not _artifact_matches(result, artifact_filter):
                continue
            if not _quant_matches(result, quant_filter):
                continue
            if fit_filter == "fits vram" and not result.fit.startswith("fits"):
                continue
            if fit_filter == "known size" and result.fit == "unknown":
                continue
            results.append(result)
        results.sort(key=_sort_key(sort), reverse=sort not in {"repo", "updated_asc"})
        with self._lock:
            self.search_results = results
            if results:
                self.last_repo_id = results[0].repo_id
                self.last_repo_type = results[0].repo_type
        self.log(f"Sifted to {len(results)} result(s).")
        return results

    def sort_current_results(self, sort: str) -> list[SearchResult]:
        sort = _normalize_local_sort(sort)
        with self._lock:
            self.search_results = sorted(
                self.search_results,
                key=_sort_key(sort),
                reverse=sort not in {"repo", "updated_asc"},
            )
        self.log(f"Sorted current results by {sort}.")
        return self.search_results

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
            result.size_bytes = sum(item.size for item in files)
            result.weight_bytes = _model_weight_bytes(repo_type, files)
            result.size = format_bytes(result.size_bytes) if result.size_bytes else result.size
            result.weights = _weight_summary_from_files(repo_type, files) or result.weights
            result.quants = _quant_size_summary(files) or result.quants
            result = _with_fit(result, self.vram_limit_gb)
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
        allow_patterns: tuple[str, ...] | None = None,
        estimated_bytes: int | None = None,
    ) -> DownloadJob:
        repo_id = repo_id.strip()
        if not repo_id:
            raise ValueError("Repo ID is required.")
        patterns = tuple(allow_patterns or _allow_patterns(repo_type))
        if estimated_bytes is None:
            estimated_bytes = self.estimate_download_size(repo_id, repo_type, allow_patterns=patterns)
        self._assert_download_space(estimated_bytes)
        job_id = f"{repo_type}:{repo_id}:{time.time_ns()}"
        job = DownloadJob(
            job_id=job_id,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            allow_patterns=patterns,
            estimated_bytes=estimated_bytes,
        )
        with self._lock:
            self.jobs[job_id] = job
            self.last_repo_id = repo_id
            self.last_repo_type = repo_type
            self._start_queued_downloads_locked()
        estimate = f" estimated {format_bytes(estimated_bytes)}" if estimated_bytes else " size unknown"
        self.log(f"Queued HF download: {repo_type} {repo_id};{estimate}.")
        return job

    def start_file_download(
        self,
        repo_id: str,
        repo_type: RepoType,
        file_path: str,
        *,
        revision: str | None = None,
    ) -> DownloadJob:
        repo_id = repo_id.strip()
        file_path = file_path.strip()
        if not repo_id or not file_path:
            raise ValueError("Repo ID and file path are required.")
        details = self._details_for_download(repo_id, repo_type, revision=revision)
        selected = next((item for item in details.files if item.path == file_path), None)
        if selected is None:
            raise ValueError(f"{file_path} was not found in {repo_id}.")
        if selected.axolotl == "skip":
            raise ValueError(f"{file_path} is a runtime/unsupported artifact and is not queued for Axolotl.")
        patterns = _file_download_patterns(repo_type, selected.path)
        estimated = _estimate_from_files(details.files, patterns)
        return self.start_download(
            repo_id,
            repo_type,
            revision=revision,
            allow_patterns=patterns,
            estimated_bytes=estimated,
        )

    def estimate_download_size(
        self,
        repo_id: str,
        repo_type: RepoType,
        *,
        allow_patterns: tuple[str, ...],
        revision: str | None = None,
    ) -> int:
        details = self._details_for_download(repo_id, repo_type, revision=revision)
        return _estimate_from_files(details.files, allow_patterns)

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
                "Estimate": format_bytes(job.estimated_bytes) if job.estimated_bytes else "unknown",
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
        job = self.jobs[job_id]
        try:
            if job.estimated_bytes:
                self._assert_download_space(job.estimated_bytes, include_existing_reservations=False)
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            local_path = snapshot_download(
                repo_id=job.repo_id,
                repo_type=job.repo_type,
                revision=job.revision or None,
                allow_patterns=job.allow_patterns or _allow_patterns(job.repo_type),
                token=token or None,
            )
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.ended_at = time.time()
                self._start_queued_downloads_locked()
            self.log(f"Download failed for {job.repo_type} {job.repo_id}: {exc}")
            return

        with self._lock:
            job.status = "complete"
            job.local_path = local_path
            job.ended_at = time.time()
            self.last_local_path = local_path
            self._start_queued_downloads_locked()
        self.log(f"Download complete for {job.repo_type} {job.repo_id}: {local_path}")

    def _details_for_download(self, repo_id: str, repo_type: RepoType, *, revision: str | None = None) -> RepoDetails:
        if self.selected_details and self.selected_details.result.repo_id == repo_id and self.selected_details.result.repo_type == repo_type:
            return self.selected_details
        details = self.inspect_repo(repo_id, repo_type, revision=revision)
        if details is None:
            raise ValueError(f"Could not inspect {repo_type} {repo_id} before queueing download.")
        return details

    def _assert_download_space(self, estimated_bytes: int, *, include_existing_reservations: bool = True) -> None:
        if estimated_bytes <= 0:
            self.log("Download size is unknown; disk-space check is advisory only.")
            return
        cache_dir = _cache_dir()
        free = shutil.disk_usage(cache_dir).free
        required = _space_required(estimated_bytes)
        if include_existing_reservations:
            with self._lock:
                required += sum(
                    _space_required(job.estimated_bytes)
                    for job in self.jobs.values()
                    if job.status in {"queued", "running"} and job.estimated_bytes > 0
                )
        if free < required:
            raise ValueError(
                f"Not enough free space in HF cache {cache_dir}: need {format_bytes(required)}, "
                f"available {format_bytes(free)}."
            )

    def _start_queued_downloads_locked(self) -> None:
        running = sum(1 for job in self.jobs.values() if job.status == "running")
        slots = max(0, self.max_concurrent_downloads - running)
        if slots <= 0:
            return
        queued = sorted(
            (job for job in self.jobs.values() if job.status == "queued"),
            key=lambda job: job.started_at,
        )
        for job in queued[:slots]:
            job.status = "running"
            thread = threading.Thread(target=self._download_worker, args=(job.job_id,), daemon=True)
            thread.start()

    def _result_from_info(self, item: object, repo_type: RepoType) -> SearchResult:
        repo_id = str(getattr(item, "id", "") or getattr(item, "repo_id", ""))
        tags = getattr(item, "tags", None) or []
        updated = getattr(item, "last_modified", None) or getattr(item, "lastModified", None)
        siblings = _siblings(item)
        filenames = [_sibling_path(sibling) for sibling in siblings]
        classification = _classify_repo(repo_type, repo_id, filenames, tags)
        base_models = _base_model_ids(getattr(item, "base_models", None) or getattr(item, "baseModels", None))
        children_count = getattr(item, "children_model_count", None) or getattr(item, "childrenModelCount", None)
        size = _repo_size(item, siblings)
        weight_bytes = _weight_bytes(repo_type, filenames, siblings, size)
        params = _param_summary(item)
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
            size_bytes=size,
            weight_bytes=weight_bytes,
            params=params,
            role=classification["role"],
            weights=classification["weights"],
            quants=classification["quants"],
            fit=_fit_label(weight_bytes, self.vram_limit_gb),
            compatibility=classification["compatibility"],
            blocked=classification["blocked"] == "true",
        )

    def _token(self) -> str | bool | None:
        return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None

    def _list_models(self, *, query: str, sort: str, limit: int) -> object:
        try:
            return list(
                self.api.list_models(
                    search=query,
                    sort=sort,
                    limit=limit,
                    expand=[
                        "baseModels",
                        "downloads",
                        "gated",
                        "gguf",
                        "lastModified",
                        "library_name",
                        "likes",
                        "pipeline_tag",
                        "safetensors",
                        "sha",
                        "siblings",
                        "tags",
                        "transformersInfo",
                    ],
                    token=self._token(),
                )
            )
        except Exception:
            return list(
                self.api.list_models(
                    search=query,
                    sort=sort,
                    limit=limit,
                    full=True,
                    token=self._token(),
                )
            )

    def _list_datasets(self, *, query: str, sort: str, limit: int) -> object:
        try:
            return list(
                self.api.list_datasets(
                    search=query,
                    sort=sort,
                    limit=limit,
                    expand=[
                        "downloads",
                        "gated",
                        "lastModified",
                        "likes",
                        "sha",
                        "siblings",
                        "tags",
                    ],
                    token=self._token(),
                )
            )
        except Exception:
            return list(
                self.api.list_datasets(
                    search=query,
                    sort=sort,
                    limit=limit,
                    full=True,
                    token=self._token(),
                )
            )


def search_rows(results: list[SearchResult]) -> list[dict[str, str]]:
    if not results:
        return [{"Repo": "No results yet", "Fit": "", "Weights / Quants": "", "Files": "", "Downloads": "", "Likes": "", "Updated": ""}]
    return [
        {
            "Repo": item.repo_id,
            "Fit": item.fit,
            "Weights / Quants": _compact_size_cell(item),
            "Files": str(item.file_count or ""),
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


def result_link_markdown(results: list[SearchResult], details: RepoDetails | None = None) -> str:
    lines = [
        "| [Repo](/hf/sort/repo) | Fit | Weights / Quants | Files | [Downloads](/hf/sort/downloads) | [Likes](/hf/sort/likes) | [Updated](/hf/sort/updated) |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    if not results:
        lines.append("| No results loaded |  |  |  |  |  |  |")
    for index, item in enumerate(results[:20], start=1):
        encoded = quote(item.repo_id, safe="")
        select_href = f"/hf/select/{item.repo_type}/{encoded}"
        size = item.quants or item.weights or item.size or "inspect for sizes"
        fit = item.fit or "fit unknown"
        hf_path = _hf_path(item)
        lines.append(
            f"| {index}. [{item.repo_id}]({select_href})<br>`{hf_path}` | {fit} | {size} | "
            f"{item.file_count or ''} | {'' if item.downloads is None else f'{item.downloads:,}'} | "
            f"{'' if item.likes is None else f'{item.likes:,}'} | {item.updated} |"
        )
        if details and details.result.repo_id == item.repo_id and details.result.repo_type == item.repo_type:
            lines.extend(_expanded_repo_markdown(details))
    if details and not any(item.repo_id == details.result.repo_id and item.repo_type == details.result.repo_type for item in results):
        lines.append("")
        lines.append(f"**Expanded selection:** `{details.result.repo_id}`")
        lines.extend(_expanded_repo_markdown(details))
    return "\n".join(lines)


def _expanded_repo_markdown(details: RepoDetails) -> list[str]:
    result = details.result
    lines = [
        "",
        f"**Expanded:** `{result.repo_id}` - {result.compatibility}",
        "",
        "| File / Quant | Size | Kind | Axolotl | Queue |",
        "| --- | ---: | --- | --- | --- |",
    ]
    files = sorted(details.files, key=lambda item: (item.axolotl == "skip", item.kind, item.path.lower()))
    for item in files[:40]:
        encoded_repo = quote(result.repo_id, safe="")
        encoded_file = quote(item.path, safe="")
        queue = "blocked" if item.axolotl == "skip" else f"[queue](/hf/download-file/{result.repo_type}/{encoded_repo}?file={encoded_file})"
        lines.append(
            f"| `{item.path}` | {format_bytes(item.size) if item.size else 'unknown'} | {item.kind} | {item.axolotl} | {queue} |"
        )
    if len(files) > 40:
        lines.append(f"| ... {len(files) - 40} more file(s) |  |  |  | inspect via Selection panel |")
    return lines


def detail_summary_rows(details: RepoDetails | None) -> list[dict[str, str]]:
    if details is None:
        return [{"Field": "Selected", "Value": "No repository inspected yet"}]
    result = details.result
    rows = [
        ("Repo ID", result.repo_id),
        ("Type", result.repo_type),
        ("Use As", _use_as(result)),
        ("Compatibility", result.compatibility),
        ("VRAM Fit", result.fit),
        ("Size", result.size),
        ("Parameters", result.params),
        ("Weights / Files", result.weights),
        ("Quant / Weight Sizes", result.quants),
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
        return [{"Repo": "No related fine-tunes loaded", "Fit": "", "Weights / Quants": "", "Downloads": ""}]
    return [
        {
            "Repo": item.repo_id,
            "Fit": item.fit,
            "Weights / Quants": _compact_size_cell(item),
            "Downloads": "" if item.downloads is None else f"{item.downloads:,}",
        }
        for item in results
    ]


def cache_summary_text(total_bytes: int, total_text: str) -> str:
    return total_text if total_text != "0B" else format_bytes(total_bytes)


def _compact_size_cell(item: SearchResult) -> str:
    return item.quants or item.weights or item.size or item.params or "inspect"


def _hf_path(item: SearchResult) -> str:
    prefix = "datasets/" if item.repo_type == "dataset" else ""
    return f"huggingface.co/{prefix}{item.repo_id}"


def _use_as(item: SearchResult) -> str:
    if item.repo_type == "dataset":
        return "dataset path"
    if item.role == "peft_adapter":
        return "lora_model_dir"
    if item.role == "runtime_quant":
        return "blocked runtime artifact"
    return "base_model"


def _normalize_sort(sort: str) -> str:
    value = (sort or "downloads").strip()
    return value if value in {"downloads", "likes", "last_modified", "trending_score"} else "downloads"


def _normalize_local_sort(sort: str) -> str:
    value = (sort or "downloads").strip()
    return value if value in {"downloads", "likes", "repo", "updated", "updated_asc", "size", "fit"} else "downloads"


def _sort_key(sort: str):
    def key(item: SearchResult) -> object:
        if sort == "likes":
            return item.likes or 0
        if sort == "repo":
            return item.repo_id.lower()
        if sort in {"updated", "updated_asc"}:
            return item.updated
        if sort == "size":
            return item.weight_bytes or item.size_bytes
        if sort == "fit":
            return _fit_sort_value(item)
        return item.downloads or 0

    return key


def _fit_sort_value(item: SearchResult) -> float:
    if item.fit.startswith("fits"):
        return 2.0
    if item.fit.startswith("too large"):
        return 0.0
    return 1.0


def _artifact_matches(item: SearchResult, artifact_filter: str) -> bool:
    value = (artifact_filter or "any").strip().lower()
    if value in {"any", "any artifact"}:
        return True
    haystack = _filter_haystack(item)
    if value in {"trainable models", "base/trainable models"}:
        return item.repo_type == "model" and not item.blocked and item.role not in {"peft_adapter", "runtime_quant"}
    if value in {"adapters", "peft adapters"}:
        return "adapter" in haystack
    if value == "runtime only":
        return any(token in haystack for token in ("gguf", "runtime", "onnx", "exl2"))
    if value == "datasets":
        return item.repo_type == "dataset"
    return True


def _quant_matches(item: SearchResult, quant_filter: str) -> bool:
    value = (quant_filter or "any").strip().lower()
    if value in {"any", "any weight format"} or item.repo_type == "dataset":
        return True
    haystack = _filter_haystack(item)
    checks = {
        "safetensors": ("safetensors",),
        "transformers safetensors": ("safetensors",),
        "bf16/fp16": ("bf16", "fp16", "float16", "bfloat16"),
        "full precision fp16/bf16": ("bf16", "fp16", "float16", "bfloat16"),
        "4-bit": ("4-bit", "4bit", "int4", "q4", "gptq", "awq"),
        "4-bit quantized": ("4-bit", "4bit", "int4", "q4", "gptq", "awq"),
        "8-bit": ("8-bit", "8bit", "int8", "q8"),
        "8-bit quantized": ("8-bit", "8bit", "int8", "q8"),
        "gptq": ("gptq",),
        "gptq quantized": ("gptq",),
        "awq": ("awq",),
        "awq quantized": ("awq",),
        "gguf": ("gguf",),
        "gguf runtime files": ("gguf",),
    }
    return any(token in haystack for token in checks.get(value, ()))


def _filter_haystack(item: SearchResult) -> str:
    return f"{item.role} {item.weights} {item.quants} {item.compatibility} {item.tags} {item.repo_id}".lower()


def _with_fit(item: SearchResult, vram_limit_gb: float | None) -> SearchResult:
    item.fit = _fit_label(item.weight_bytes or item.size_bytes, vram_limit_gb)
    return item


def _fit_label(size_bytes: int, vram_limit_gb: float | None) -> str:
    if not vram_limit_gb:
        return "set VRAM"
    if size_bytes <= 0:
        return "unknown"
    limit = vram_limit_gb * BYTES_PER_GIB
    if size_bytes <= limit:
        return f"fits {vram_limit_gb:g}GB"
    return f"too large for {vram_limit_gb:g}GB"


def _allow_patterns(repo_type: RepoType) -> tuple[str, ...]:
    return MODEL_DOWNLOAD_ALLOW if repo_type == "model" else DATASET_DOWNLOAD_ALLOW


def _file_download_patterns(repo_type: RepoType, file_path: str) -> tuple[str, ...]:
    if repo_type == "dataset":
        return (*DATASET_SUPPORT_DOWNLOAD_ALLOW, file_path)
    return (*MODEL_SUPPORT_DOWNLOAD_ALLOW, file_path)


def _estimate_from_files(files: list[RepoFile], allow_patterns: tuple[str, ...]) -> int:
    return sum(item.size for item in files if _pattern_allowed(item.path, allow_patterns))


def _pattern_allowed(path: str, allow_patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in allow_patterns)


def _cache_dir() -> Path:
    raw = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE") or HF_HUB_CACHE
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _space_required(estimated_bytes: int) -> int:
    return int(estimated_bytes * DOWNLOAD_SPACE_BUFFER)


def _download_concurrency() -> int:
    raw = os.getenv("HF_MAX_CONCURRENT_DOWNLOADS", "2")
    try:
        value = int(raw)
    except ValueError:
        return 2
    return max(1, min(value, 8))


def _siblings(item: object) -> list[object]:
    siblings = getattr(item, "siblings", None) or []
    return list(siblings) if isinstance(siblings, (list, tuple)) else []


def _base_model_ids(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        models = value.get("models")
        if isinstance(models, list):
            ids = []
            for model in models:
                if isinstance(model, dict):
                    model_id = model.get("id")
                    if model_id:
                        ids.append(str(model_id))
                elif model:
                    ids.append(str(model))
            return ids
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


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
    return sum(_sibling_size(sibling) for sibling in siblings)


def _param_summary(item: object) -> str:
    safetensors = getattr(item, "safetensors", None)
    total = getattr(safetensors, "total", None)
    if not isinstance(total, (int, float)) and isinstance(safetensors, dict):
        total = safetensors.get("total")
    if not isinstance(total, (int, float)) or total <= 0:
        return ""
    return _format_params(int(total))


def _format_params(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    return f"{value:,}"


def _weight_bytes(repo_type: RepoType, filenames: list[str], siblings: list[object], repo_size: int) -> int:
    if repo_type == "dataset":
        return repo_size
    total = 0
    for filename, sibling in zip(filenames, siblings, strict=False):
        lower = filename.lower()
        if lower.endswith(MODEL_WEIGHT_SUFFIXES) or lower.endswith(RUNTIME_MODEL_SUFFIXES):
            total += _sibling_size(sibling)
    return total or repo_size


def _model_weight_bytes(repo_type: RepoType, files: list[RepoFile]) -> int:
    if repo_type == "dataset":
        return sum(item.size for item in files if item.path.lower().endswith(DATASET_SUFFIXES))
    return sum(
        item.size
        for item in files
        if item.path.lower().endswith(MODEL_WEIGHT_SUFFIXES + RUNTIME_MODEL_SUFFIXES)
    )


def _weight_summary_from_files(repo_type: RepoType, files: list[RepoFile]) -> str:
    if repo_type == "dataset":
        count = sum(1 for item in files if item.path.lower().endswith(DATASET_SUFFIXES))
        return f"{count} compatible data file(s)" if count else ""
    hf_count = sum(1 for item in files if item.path.lower().endswith(MODEL_WEIGHT_SUFFIXES))
    runtime_count = sum(1 for item in files if item.path.lower().endswith(RUNTIME_MODEL_SUFFIXES))
    parts = []
    if hf_count:
        parts.append(f"{hf_count} HF weight file(s)")
    if runtime_count:
        parts.append(f"{runtime_count} runtime file(s)")
    return ", ".join(parts)


def _quant_size_summary(files: list[RepoFile]) -> str:
    groups: dict[str, int] = {}
    for item in files:
        lower = item.path.lower()
        if lower.endswith(".gguf"):
            key = _gguf_quant_label(lower)
        elif lower.endswith(".safetensors"):
            key = "safetensors"
        elif lower.endswith((".bin", ".pt")):
            key = lower.rsplit(".", 1)[-1]
        else:
            continue
        groups[key] = groups.get(key, 0) + item.size
    if not groups:
        return ""
    return ", ".join(f"{key}: {format_bytes(value)}" for key, value in sorted(groups.items()))


def _gguf_quant_label(filename: str) -> str:
    match = re.search(r"\b(q[2-8](?:_[a-z0-9]+)*)\b", filename)
    return match.group(1).upper() if match else "GGUF"


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
