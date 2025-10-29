"""
search_agent_v2: Alignment-only reward extension for search-agent geolocation.

Based on utils/reward_score/search_agent.py (base score), plus an alignment reward:
- If any search result payload mentions the target city and the model's final
  answer is city-correct → +1.0
- If any search result payload mentions the target city but the final answer is
  not city-correct → -1.0

All helper parsing remains local to this file, avoiding side effects.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .utils_geo import eval_geolocation_response
from .utils import print_hl, print_error


# -------- Helpers: extract search results from conversation text -------- #

_TOOL_RESP_RE = re.compile(r"<tool_response>([\s\S]*?)</tool_response>", re.IGNORECASE)


def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    if not text:
        return objs
    s = text
    n = len(s)
    i = 0
    while i < n:
        if s[i] == '{':
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                ch = s[j]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                j += 1
            if depth == 0:
                cand = s[i:j]
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict):
                        objs.append(obj)
                except Exception:
                    pass
                i = j
                continue
        i += 1
    return objs


def _extract_tool_responses(text: str) -> List[str]:
    return [m.group(1) for m in _TOOL_RESP_RE.finditer(text or "")]


def _extract_search_result_payloads(text: str) -> List[Dict[str, Any]]:
    """Return objects shaped like {"name":"search_web", "query":..., "result": ...}
    that appear inside <tool_response> blocks.
    """
    payloads: List[Dict[str, Any]] = []
    for seg in _extract_tool_responses(text):
        for obj in _extract_json_objects(seg):
            try:
                if obj.get("name") == "search_web" and "result" in obj:
                    payloads.append(obj)
            except Exception:
                continue
    return payloads


def _contains_ci(haystack: str, needle: str) -> bool:
    try:
        return needle.lower() in haystack.lower()
    except Exception:
        return needle in haystack


def _search_results_contain_target_city(payloads: List[Dict[str, Any]], ground_truth: Dict[str, Any]) -> bool:
    city = (ground_truth.get("city") or "").strip()
    if not city:
        return False
    # Concatenate payloads into a single string and check substring (case-insensitive)
    try:
        buf = "\n".join(json.dumps(p, ensure_ascii=False) for p in payloads)
    except Exception:
        buf = "\n".join(str(p) for p in payloads)
    return _contains_ci(buf, city)


# ------------------------------
# Main reward computation
# ------------------------------

def compure_score_search(predict_str: str, ground_truth: dict, extra_info=None) -> Dict[str, Any]:
    """Base geolocation score + alignment reward.

    Returns a dict with keys:
    - score: 0/1/2/4 base correctness (city/state/country)
    - city_correct/state_correct/country_correct: floats (0.0/1.0)
    - alignment_reward: +/- reward based on search results vs final city correctness
    - search_contains_target_city: 1.0 if target city appears in search results, else 0.0
    """
    raw_text = predict_str or ""

    # Evaluate final answer correctness (use last assistant chunk as in v0)
    last_predict_str = raw_text.split('assistant')[-1].strip()
    combined = eval_geolocation_response(
        last_predict_str,
        ground_truth,
        model_verifier=True,
        api_key='307ad8bb-2c44-4911-85e8-48ea32f6a672',
    )

    if combined['city_correct']:
        base_score = 4.0
    elif combined['state_correct']:
        base_score = 2.0
    elif combined['country_correct']:
        base_score = 1.0
    else:
        base_score = 0.0

    # Alignment reward
    payloads = _extract_search_result_payloads(raw_text)
    search_has_target_city = _search_results_contain_target_city(payloads, ground_truth)
    if search_has_target_city and combined['city_correct']:
        align_reward = 1.0
    elif search_has_target_city and not combined['city_correct']:
        align_reward = -1.0
    else:
        align_reward = 0.0

    score = base_score + align_reward

    return {
        'score': score,
        'base_score': base_score,
        'city_correct': 1.0 if combined.get('city_correct') else 0.0,
        'state_correct': 1.0 if combined.get('state_correct') else 0.0,
        'country_correct': 1.0 if combined.get('country_correct') else 0.0,
        'alignment_reward': float(align_reward),
        'search_contains_target_city': 1.0 if search_has_target_city else 0.0,
    }


if __name__ == "__main__":
    fake_conv = (
        "<|im_start|>assistant\n"
        "<tool_response>For search, the search results are as follows:\n"
        "{\n  \"name\": \"search_web\",\n  \"query\": \"Sunglass Hut near Paris\",\n  \"result\": [{\"title\": \"Store Paris\", \"snippet\": \"...Paris...\"}]\n}\n"
        "</tool_response>\n"
        "Final guess: Paris, France."
    )
    gt = {"country": "France", "province_or_state": None, "city": "Paris"}
    out = compure_score_search(fake_conv, gt)
    print_hl(json.dumps(out, ensure_ascii=False))

