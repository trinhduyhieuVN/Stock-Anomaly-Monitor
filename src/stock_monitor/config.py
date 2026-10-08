"""Load the project configuration without depending on the current directory."""

from datetime import date
from pathlib import Path

import yaml

CONFIRMATIONS = (
    "source_bar_label", "source_price_unit", "source_volume_unit",
    "price_adjustment", "data_usage", "session_calendar",
)


class ConfigError(ValueError):
    """A missing or contradictory project setting."""


def load_config(path):
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict) or settings.get("schema_version") != 1:
        raise ConfigError("Expected a schema_version: 1 configuration")
    project, data = settings.get("project"), settings.get("data")
    if not isinstance(project, dict) or not isinstance(data, dict):
        raise ConfigError("Missing project or data configuration")
    if data.get("source") != "VCI" or data.get("interval") != "5m" or data.get("source_interval") != "1m":
        raise ConfigError("This milestone supports VCI 1m -> 5m only; change the contract before changing source/interval")
    symbols = data.get("symbols")
    if not isinstance(symbols, list) or len(symbols) != 3 or not all(isinstance(s, str) and s.isalnum() and s == s.upper() for s in symbols) or len(set(symbols)) != 3:
        raise ConfigError("Expected three distinct uppercase symbols")
    if data.get("timestamp_timezone") != "UTC" or data.get("timestamp_convention") != "bar_end":
        raise ConfigError("Standard timestamps must be UTC bar ends")
    if data.get("price_unit") != "VND" or data.get("volume_unit") != "shares":
        raise ConfigError("Standard units must be VND/shares; raw units need separate confirmation")
    policy = data.get("quality_policy", {})
    expected_policy = {"auctions": "exclude", "incomplete_minutes": "reject_and_break_segment",
        "volume_mismatch": "reject_and_break_segment", "missing_bars": "do_not_fill",
        "exact_duplicates": "deduplicate_with_log", "conflicting_duplicates": "reject_conflicting_key",
        "normalize_before_confirmations": False}
    if any(policy.get(k) != value for k, value in expected_policy.items()):
        raise ConfigError("Unsupported quality policy; normalization before source confirmation is forbidden")
    try:
        if date.fromisoformat(data["start"]) > date.fromisoformat(data["end"]):
            raise ConfigError("Start date is after end date")
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError("Invalid start/end dates") from exc
    for key in ("probe_manifest", "raw_dir", "processed_dir", "report_dir"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ConfigError(f"Missing data.{key}")
    entries = data.get("confirmations")
    if not isinstance(entries, dict) or any(k not in entries for k in CONFIRMATIONS):
        raise ConfigError("Every source confirmation must be explicitly declared")
    for key in CONFIRMATIONS:
        entry = entries[key]
        if not isinstance(entry, dict) or entry.get("status") not in {"pending", "confirmed"}:
            raise ConfigError(f"Invalid confirmation status: {key}")
        if entry["status"] == "confirmed" and (entry.get("value") is None or not isinstance(entry.get("evidence"), str) or not entry["evidence"].strip()):
            raise ConfigError(f"Confirmation {key} requires a value and evidence")
    mode = data.get("normalization_mode", "strict")
    if mode not in {"strict", "research"}:
        raise ConfigError("normalization_mode must be strict or research")
    if mode == "research":
        profile = data.get("research_profile", {})
        if (profile.get("authorized_assumptions") is not True
                or profile.get("scope") != "local_research"
                or not isinstance(profile.get("limitations"), str) or not profile["limitations"].strip()
                or not isinstance(profile.get("evidence"), dict) or not profile["evidence"]
                or any(not isinstance(v, str) or not v.strip() for v in profile["evidence"].values())):
            raise ConfigError("Research normalization requires explicit authorization, evidence, local scope and limitations")
    root_setting = project.get("root", "..")
    if not isinstance(root_setting, str) or not root_setting:
        raise ConfigError("project.root must be a nonempty path string")
    settings["_config_path"] = config_path
    settings["_root"] = (config_path.parent / root_setting).resolve()
    for key in ("probe_manifest", "raw_dir", "processed_dir", "report_dir"):
        project_path(settings, key)
    return settings


def project_path(settings, key):
    root = settings["_root"]
    path = (root / settings["data"][key]).resolve()
    if not path.is_relative_to(root):
        raise ConfigError(f"data.{key} must stay inside the project root")
    return path


def pending_confirmations(settings):
    return [key for key in CONFIRMATIONS if settings["data"]["confirmations"][key]["status"] != "confirmed"]
