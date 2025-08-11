import os
import sys
import json
from tqdm import tqdm
from datasets import Dataset
from typing import Any


from evalplus.data import get_mbpp_plus, get_human_eval_plus, get_mbpp_plus_hash, get_human_eval_plus_hash
from evalplus.sanitize import sanitize
from evalplus.eval import untrusted_check, MBPP_OUTPUT_NOT_NONE_TASKS, estimate_pass_at_k, PASS
from evalplus.evaluate import get_groundtruth


CACHE_KEYS = [
    ("HumanEval/83", "plus"),
    ("HumanEval/139", "plus"),   
]

CACHE_SIGNAL = "CACHED"


def prepare_for_json(obj):
    if isinstance(obj, dict):
        if any(not isinstance(key, str) for key in obj.keys()):
            return {
                "__dict_with_non_str_keys__": True,
                "data": [[prepare_for_json(k), prepare_for_json(v)] for k, v in obj.items()]
            }
        return {k: prepare_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [prepare_for_json(elem) for elem in obj]
    
    if isinstance(obj, tuple):
        return {"__tuple__": True, "items": [prepare_for_json(elem) for elem in obj]}

    if isinstance(obj, set):
        return {"__set__": True, "items": [prepare_for_json(elem) for elem in obj]}

    if isinstance(obj, complex):
        return {"__complex__": True, "real": obj.real, "imag": obj.imag}
    
    return obj


def advanced_decoder(dct):
    if not isinstance(dct, dict):
        return dct

    if "__dict_with_non_str_keys__" in dct:
        return {advanced_decoder(item[0]): advanced_decoder(item[1]) for item in dct["data"]}
    if "__tuple__" in dct:
        return tuple(advanced_decoder(item) for item in dct["items"])
    if "__set__" in dct:
        return {advanced_decoder(item) for item in dct["items"]}
    if "__complex__" in dct:
        return complex(dct["real"], dct["imag"])
    
    return dct


def serialize_value(v: Any) -> Dataset:
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(prepare_for_json(v))
        except Exception as e:
            raise ValueError(f"Failed to serialize value: {v} because of {e}")
    return v


def prepare_mbpp(**kwargs) -> dict[str, Dataset]:
    dataset = get_mbpp_plus()
    dataset_hash = get_mbpp_plus_hash()
    expected_output = get_groundtruth(
        dataset,
        dataset_hash,
        MBPP_OUTPUT_NOT_NONE_TASKS,
    )
    assert len(dataset) == len(expected_output), f"Dataset length mismatch: {len(dataset)} != {len(expected_output)}"
    return {"test": convert_to_dataset(dataset, expected_output)}


def prepare_humaneval(**kwargs) -> dict[str, Dataset]:
    dataset = get_human_eval_plus()
    dataset_hash = get_human_eval_plus_hash()
    expected_output = get_groundtruth(dataset, dataset_hash, [])
    assert len(dataset) == len(expected_output), f"Dataset length mismatch: {len(dataset)} != {len(expected_output)}"
    return {"test": convert_to_dataset(dataset, expected_output)}


def convert_to_dataset(dataset: dict, expected_output: dict) -> dict[str, Dataset]:
    first_sample_key = list(dataset.keys())[0]
    
    dataset_dict = {}
    for key in dataset[first_sample_key].keys():
        dataset_dict[key] = []
    for key in expected_output[first_sample_key].keys():
        dataset_dict["expected_output_" + key] = []

    for key in tqdm(list(dataset.keys()), desc="Converting dataset to dict"):
        for sub_key in dataset[key].keys():
            dataset_dict[sub_key].append(serialize_value(dataset[key][sub_key]))
        for sub_key in expected_output[key].keys():
            try:
                if (key, sub_key) in CACHE_KEYS:
                    # this sample contains a very large int number, which is inconvenient to be converted to a string
                    dataset_dict["expected_output_" + sub_key].append(CACHE_SIGNAL)
                else:
                    dataset_dict["expected_output_" + sub_key].append(
                        serialize_value(expected_output[key][sub_key]))
            except Exception as e:
                raise ValueError(
                    f"Failed to convert dataset key {key} sub_key {sub_key}: {e}"
                )
    
    dataset_dict = Dataset.from_dict(dataset_dict, split="test")

    return dataset_dict


def postprocess(resps: list[list[str]], docs: list[dict]) -> str:
    processed_resps = []
    for resp, doc in zip(resps, docs):
        processed_resp = []
        for _resp in resp:
            sanitized_code = sanitize(_resp, doc["entry_point"])
            processed_resp.append(sanitized_code)
        processed_resps.append(processed_resp)

    return processed_resps


def pass_at_1(references: list[str], predictions: list[str], dataset: str, mode: str):
    base_inp, plus_inp, entry_point, expected_base, expected_plus, \
        atol, ref_time_base, ref_time_plus = references[0]
    solution = predictions[0][0]

    res_base = untrusted_check(
        dataset,
        solution,
        base_inp,
        entry_point,
        expected_base,
        atol,
        ref_time_base,
        fast_check=True,
    )

    if mode == "plus":
        res_plus = untrusted_check(
            dataset,
            solution,
            plus_inp,
            entry_point,
            expected_plus,
            atol,
            ref_time_plus,
            fast_check=True,
        )
        is_correct = (res_base[0] == res_plus[0] == PASS)
    else:
        is_correct = (res_base[0] == PASS)
    
    pass_at_k = estimate_pass_at_k(1, [is_correct], 1)[0]

    return pass_at_k


def pass_at_1_base(references, predictions, dataset):
    return pass_at_1(references, predictions, dataset, "base")

def pass_at_1_plus(references, predictions, dataset):
    return pass_at_1(references, predictions, dataset, "plus")


def doc_to_target(doc: dict) -> tuple:
    if doc["expected_output_plus"] == CACHE_SIGNAL:
        dataset = get_human_eval_plus()
        dataset_hash = get_human_eval_plus_hash()
        expected_output = get_groundtruth(dataset, dataset_hash, [])
        expected_output_plus = expected_output[doc["task_id"]]["plus"]
    else:
        expected_output_plus = json.loads(doc["expected_output_plus"], object_hook=advanced_decoder)

    return (
        json.loads(doc["base_input"], object_hook=advanced_decoder),
        json.loads(doc["plus_input"], object_hook=advanced_decoder),
        doc["entry_point"],
        json.loads(doc["expected_output_base"], object_hook=advanced_decoder),
        expected_output_plus,
        doc["atol"],
        json.loads(doc["expected_output_base_time"], object_hook=advanced_decoder),
        json.loads(doc["expected_output_plus_time"], object_hook=advanced_decoder),
    )