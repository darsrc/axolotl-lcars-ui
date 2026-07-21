"""Hugging Face browsing, download, and cache-management helpers."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download

from axolotl_lcars_ui.resources import format_bytes


RepoType = Literal["model", "dataset"]


@dataclass
class SearchResult:
    repo_id: str
    repo_type: RepoType
    downloads: int | None = None
    likes: int | None = None
    tags: str = ""
    updated: str = ""


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
        self.jobs: dict[str, DownloadJob] = {}
        self.logs: deque[str] = deque(maxlen=300)
        self.last_repo_id = ""
        self.last_repo_type: RepoType = "model"
        self.last_local_path = ""
        self._lock = threading.Lock()

    def search(self, query: str, repo_type: RepoType, *, limit: int = 12) -> list[SearchResult]:
        query = query.strip()
        if not query:
            self.log("Search skipped: empty query.")
            return self.search_results
        self.log(f"Searching Hugging Face {repo_type}s for {query!r}.")
        try:
            if repo_type == "model":
                items = self.api.list_models(search=query, sort="downloads", limit=limit)
            else:
                items = self.api.list_datasets(search=query, sort="downloads", limit=limit)
            results = [self._result_from_info(item, repo_type) for item in items]
        except Exception as exc:  # network/auth/client errors are user-visible in the log
            self.log(f"HF search failed: {exc}")
            return self.search_results

        with self._lock:
            self.search_results = results
            if results:
                self.last_repo_id = results[0].repo_id
                self.last_repo_type = repo_type
        self.log(f"Search returned {len(results)} {repo_type} result(s).")
        return results

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
        return SearchResult(
            repo_id=repo_id,
            repo_type=repo_type,
            downloads=getattr(item, "downloads", None),
            likes=getattr(item, "likes", None),
            tags=", ".join(str(tag) for tag in tags[:5]),
            updated="" if updated is None else str(updated)[:19],
        )


def search_rows(results: list[SearchResult]) -> list[dict[str, str]]:
    if not results:
        return [{"Repo": "No results yet", "Type": "", "Downloads": "", "Likes": "", "Tags": ""}]
    return [
        {
            "Repo": item.repo_id,
            "Type": item.repo_type,
            "Downloads": "" if item.downloads is None else f"{item.downloads:,}",
            "Likes": "" if item.likes is None else f"{item.likes:,}",
            "Tags": item.tags[:80],
        }
        for item in results
    ]


def cache_summary_text(total_bytes: int, total_text: str) -> str:
    return total_text if total_text != "0B" else format_bytes(total_bytes)
