# eval/multi_level_eval_gemini_0825.py
import json
import re
import unicodedata
from typing import Dict, Optional, Tuple, Any

# from utils_gemini import chat_gemini
# from .utils_gpt import chat_4o_mini
from .utils_api import chat_4o_mini
from .utils import print_hl, print_error


def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _contains_whole_term(text_norm: str, term_norm: str) -> bool:
    if not term_norm:
        return False
    pattern = r"(?<!\w)" + re.escape(term_norm) + r"(?!\w)"
    return re.search(pattern, text_norm) is not None


def _extract_json_obj(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s = s.strip()
    # Try direct parse first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback: extract first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = s[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def eval_geolocation_response(
    response: str,
    loc_dict: Dict[str, Optional[str]],
    model_verifier: bool = False,
    api_key: Optional[str] = None,
    timeout: int = 120,
    debug_mode: bool = False,
) -> Dict[str, Any]:
    """
    评测模型回答的地理定位是否与 loc_dict 一致。
    流程：
    1) 先进行本地规则匹配（rule-based）得到 rb = {country_correct, state_correct, city_correct}。
    2) 若 model_verifier=True，则调用 Gemini 进行一次判断，得到 mb，同样三个布尔值；
       然后与 rb 做按位“或”合并：combined = rb OR mb。
    3) 对 combined 做阶梯式一致性修正：
       - 若 city_correct=True，则同时令 state_correct=True, country_correct=True。
       - 否则若 state_correct=True，则令 country_correct=True。
    4) 返回修正后的 combined。
    """


    # ------ 规则匹配（本地）------
    def rule_based() -> Dict[str, bool]:
        res_norm = _normalize_text(response or "")
        city_norm = _normalize_text(loc_dict.get("city"))
        state_norm = _normalize_text(
            loc_dict.get("province_or_state") or loc_dict.get("state")
        )
        country_norm = _normalize_text(loc_dict.get("country"))

        city_hit = _contains_whole_term(res_norm, city_norm) if city_norm else False
        state_hit = _contains_whole_term(res_norm, state_norm) if state_norm else False
        country_hit = _contains_whole_term(res_norm, country_norm) if country_norm else False

        city_correct = city_hit
        state_correct = state_hit or city_hit
        country_correct = country_hit or state_hit or city_hit

        return {
            "country_correct": bool(country_correct),
            "state_correct": bool(state_correct),
            "city_correct": bool(city_correct),
        }

    rb = rule_based()  # 1) 始终先做本地规则匹配
    # print_hl("[eval_geolocation_response] rule based raw response:")
    # print(rb if isinstance(rb, str) else json.dumps(rb, ensure_ascii=False))

    # 若不启用模型裁判，直接做阶梯一致性并返回
    if not model_verifier:
        combined = dict(rb)
        # 3) 阶梯式一致性修正
        if combined["city_correct"]:
            combined["state_correct"] = True
            combined["country_correct"] = True
        elif combined["state_correct"]:
            combined["country_correct"] = True
        return combined

    # ------ 模型裁判（一次调用）------
    def _model_verdict() -> Optional[Dict[str, bool]]:
        try:
            sys_prompt = (
                "You are a strict evaluator. Decide if a free-text geolocation answer matches a gold location.\n"
                "Rules:\n"
                "1) If the answer names the correct city (as a toponym), then city/state/country are all True.\n"
                "2) If it names the correct state/province (but not the correct city), then state and country are True; city is False.\n"
                "3) If it names only the correct country, then only country is True.\n"
                "4) If none match, all are False.\n"
                "5) Consider common synonyms and English exonyms; ignore punctuation and case.\n"
                "Respond with a single JSON object: {\"country_correct\": <bool>, \"state_correct\": <bool>, \"city_correct\": <bool>}."
            )
            user_payload = {
                "gold_location": {
                    "country": loc_dict.get("country"),
                    "province_or_state": loc_dict.get("province_or_state") or loc_dict.get("state"),
                    "city": loc_dict.get("city"),
                },
                "model_response": response,
            }
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}]},
            ]
            resp = chat_4o_mini(messages, timeout=timeout, api_key=api_key)
            if debug_mode and model_verifier:
                try:
                    print_hl("[eval_geolocation_response] model_verifier raw response:")
                    print(resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False))
                except Exception:
                    pass
            obj = _extract_json_obj(resp)
            if isinstance(obj, dict):
                print_hl(f"[eval_geolocation_response] model based raw response: {json.dumps(obj, ensure_ascii=False)=}")
                return {
                    "country_correct": bool(obj.get("country_correct") is True),
                    "state_correct": bool(obj.get("state_correct") is True),
                    "city_correct": bool(obj.get("city_correct") is True),
                }
            else:
                raise ValueError(f'Could not extract json object from raw response: {resp}')
        except Exception as e:
            print_error(f"[eval_geolocation_response] model_verifier exception: {e}")
            return rb

    mb = _model_verdict()

    # 2) model verifier 为准
    combined = mb

    # 3) 阶梯式一致性修正
    if combined["city_correct"]:
        combined["state_correct"] = True
        combined["country_correct"] = True
    elif combined["state_correct"]:
        combined["country_correct"] = True

    # 4) 返回
    return combined


def compure_score_search(predict_str: str, ground_truth: dict, extra_info=None) -> float:
    last_predict_str = predict_str.split('assistant')[-1].strip()
    # print(f'[DEBUG EVALUATION] compute_score_search \n{predict_str=} \n{ground_truth=} \n{last_predict_str=}')

    combined = eval_geolocation_response(last_predict_str, ground_truth, 
                                         model_verifier=True,
                                         api_key='307ad8bb-2c44-4911-85e8-48ea32f6a672')
    if combined['city_correct']:
        score = 4.
    elif combined['state_correct']:
        score = 2.
    elif combined['country_correct']:
        score = 1.
    else:
        score = 0.
    
    return {
        'score': score,
        'city_correct': 1.0 if combined['city_correct'] else 0.0,
        'state_correct': 1.0 if combined['state_correct'] else 0.0,
        'country_correct': 1.0 if combined['country_correct'] else 0.0,
    }


if __name__ == "__main__":
    # Lightweight tests for compure_score_search with a stubbed verifier
    # to avoid network calls. Each case overrides chat_4o_mini to return
    # a controlled JSON verdict and prints the computed score mapping.

    def run_case(name: str, stub_verdict: Dict[str, bool], predict_str: str, loc: Dict[str, str]):
        # Monkeypatch the verifier
        global chat_4o_mini
        _orig_chat = chat_4o_mini

        def _fake_chat(messages, timeout: int = 120, api_key: Optional[str] = None):
            return json.dumps(stub_verdict)

        # chat_4o_mini = _fake_chat
        try:
            res = compure_score_search(predict_str, loc)
            print_hl(f"[TEST] {name} -> {json.dumps(res, ensure_ascii=False)}")
        finally:
            chat_4o_mini = _orig_chat

    gt = {"country": "United States", "province_or_state": "California", "city": "San Francisco"}
    # Include the 'assistant' token to exercise last_predict_str extraction
    pred = "system: context...\nassistant I think it's near San Francisco, CA, USA."

    # city -> 4
    run_case(
        name="city",
        stub_verdict={"country_correct": False, "state_correct": False, "city_correct": True},
        predict_str=pred,
        loc=gt,
    )
