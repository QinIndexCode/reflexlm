from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".c",
    ".cfg",
    ".cpp",
    ".css",
    ".csv",
    ".dot",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".mmd",
    ".ps1",
    ".py",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_PATH_PREFIXES = (
    "private/",
    ".private/",
    "tmp/",
    "docs/submissions/",
    "docs/reports/",
    "artifacts/submissions/",
)
FORBIDDEN_FILENAMES = {
    "cover_letter_draft.md",
    "submission_compliance_checklist.md",
    "simulated_reviewer_action_table.md",
    "rejection_risk_audit.md",
    "create-private-github-repo-and-push.ps1",
}
ALLOWED_ABSOLUTE_PATH_FIXTURES = {
    "src/reflexlm/cli/audit_phase2m_external_generalization.py",
    "src/reflexlm/cli/audit_phase2s_open_repair.py",
    "tests/test_phase2m_external_generalization.py",
}
PERSONAL_PATTERNS = (
    re.compile("qin" + "indexcode", re.IGNORECASE),
    re.compile("qin" + r"\s+" + "zhiheng", re.IGNORECASE),
    re.compile("github" + r"\.com/" + "Qin" + "IndexCode", re.IGNORECASE),
    re.compile("gmail" + r"\.com", re.IGNORECASE),
)
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/](?:Users|MyCode)[\\/]"),
    re.compile(r"/(?:home|Users)/[^/\s]+/"),
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
)


def candidate_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(
        {
            line.strip().replace("\\", "/")
            for line in result.stdout.splitlines()
            if line.strip() and (ROOT / line.strip()).is_file()
        }
    )


def audit() -> list[str]:
    findings: list[str] = []
    for relative in candidate_paths():
        lower = relative.lower()
        name = Path(relative).name
        if lower.startswith(FORBIDDEN_PATH_PREFIXES):
            findings.append(f"forbidden public path: {relative}")
            continue
        if name in FORBIDDEN_FILENAMES:
            findings.append(f"forbidden workflow filename: {relative}")
        path = ROOT / relative
        if path.stat().st_size > 5 * 1024 * 1024:
            findings.append(f"file exceeds 5 MiB public limit: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES and name not in {
            ".gitignore",
            ".gitattributes",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"non-UTF-8 public text file: {relative}")
            continue
        for pattern in PERSONAL_PATTERNS:
            if pattern.search(text):
                findings.append(f"personal identifier in public text: {relative}")
                break
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"possible secret in public text: {relative}")
                break
        if relative not in ALLOWED_ABSOLUTE_PATH_FIXTURES:
            for pattern in ABSOLUTE_PATH_PATTERNS:
                if pattern.search(text):
                    findings.append(f"absolute machine path in public text: {relative}")
                    break
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the prospective public repository.")
    parser.parse_args()
    findings = audit()
    if findings:
        print("Public release audit failed:")
        for finding in findings:
            print(f"- {finding}")
        raise SystemExit(1)
    print(f"Public release audit passed: {len(candidate_paths())} files checked.")


if __name__ == "__main__":
    main()
