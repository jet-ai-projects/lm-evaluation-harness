# MIT License
#
# Copyright (c) 2023 THU-KEG & Zhipu AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import string
from collections import Counter

try:
    import jieba
    from fuzzywuzzy import fuzz
    from rouge import Rouge
except ImportError:
    raise ImportError(
        'Please install the required dependencies for this task with `pip install lm_eval["longbench"] or `pip install jieba fuzzywuzzy rouge`'
    )

# taken from https://github.com/THUDM/LongBench


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# def normalize_zh_answer(s: str) -> str:
#     """Lower text and remove punctuation, extra whitespace."""

#     def white_space_fix(text):
#         return "".join(text.split())

#     def remove_punc(text):
#         cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
#         all_punctuation = set(string.punctuation + cn_punctuation)
#         return "".join(ch for ch in text if ch not in all_punctuation)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_punc(lower(s)))


def count_score(predictions: list[str], references: list[str], **kwargs) -> float:
    prediction= predictions[0]
    numbers = re.findall(r"\d+", prediction)
    
    score = 0.0
    for ground_truth in references:
        right_num = 0
        for number in numbers:
            if str(number) == str(ground_truth):
                right_num += 1
        final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
        score = max(score, final_score)
    return float(score)


def retrieval_score(predictions: list[str], references: list[str], **kwargs) -> float:
    prediction = predictions[0]
    
    score = 0.0
    for ground_truth in references:
        pattern = r"Paragraph (\d+)"
        matches = re.findall(pattern, ground_truth)
        ground_truth_id = matches[0]
        numbers = re.findall(r"\d+", prediction)
        right_num = 0
        for number in numbers:
            if str(number) == str(ground_truth_id):
                right_num += 1
        final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
        score = max(score, final_score)
    return float(score)


# def retrieval_zh_score(
#     predictions: list[str], references: list[str], **kwargs
# ) -> float:
#     prediction, ground_truth = predictions[0], references[0]
#     pattern = r"段落(\d+)"
#     matches = re.findall(pattern, ground_truth)
#     ground_truth_id = matches[0]
#     numbers = re.findall(r"\d+", prediction)
#     right_num = 0
#     for number in numbers:
#         if str(number) == str(ground_truth_id):
#             right_num += 1
#     final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
#     return float(final_score)


def code_sim_score(predictions: list[str], references: list[str], **kwargs) -> float:
    prediction = predictions[0]
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    score = 0.0
    for ground_truth in references:
        score = max(score, fuzz.ratio(prediction, ground_truth) / 100)
    
    return score


def classification_score(doc: dict, results: list[str], **kwargs) -> dict:
    prediction = results[0]
    em_match_list = []
    all_classes = doc["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    
    score = 0.0
    for ground_truth in doc["answers"]:
        for match_term in em_match_list:
            if match_term in ground_truth and match_term != ground_truth:
                em_match_list.remove(match_term)
        if ground_truth in em_match_list:
            score = max(score, 1.0 / len(em_match_list))
    return {"score": score}


def rouge_score(predictions: list[str], references: list[str], **kwargs) -> float:
    global rouge
    if "rouge" not in globals():
        rouge = Rouge()
    
    prediction = predictions[0]
    score = 0
    if len(prediction.strip()) > 0 and len(prediction.strip(".")) > 0:
        for ground_truth in references:
            score = max(score, rouge.get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"])
    return score


# def rouge_zh_score(predictions: list[str], references: list[str], **kwargs) -> float:
#     prediction, ground_truth = predictions[0], references[0]
#     prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
#     ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
#     score = rouge_score([prediction], [ground_truth])
#     return score


def f1_score(predictions: list[str], references: list[str], **kwargs) -> float:
    prediction= predictions[0]
    score = 0
    for ground_truth in references:
        common = Counter(prediction) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            break
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        score = max(score, f1)
    return score


def qa_f1_score(predictions: list[str], references: list[str], **kwargs) -> float:
    prediction = predictions[0]
    score = 0
    normalized_prediction = normalize_answer(prediction)
    prediction_tokens = normalized_prediction.split()
    for ground_truth in references:
        normalized_ground_truth = normalize_answer(ground_truth)
        ground_truth_tokens = normalized_ground_truth.split()

        score = max(score, f1_score([prediction_tokens], [ground_truth_tokens]))

    return score


def score(predictions: list[str], references: list[str], score_type, **kwargs) -> float:
    if score_type == "qa_f1_score":
        return qa_f1_score(predictions, references, **kwargs)
    elif score_type == "count_score":
        return count_score(predictions, references, **kwargs)
    elif score_type == "retrieval_score":
        return retrieval_score(predictions, references, **kwargs)
    elif score_type == "code_sim_score":
        return code_sim_score(predictions, references, **kwargs)
    elif score_type == "rouge_score":
        return rouge_score(predictions, references, **kwargs)
    elif score_type == "f1_score":
        return f1_score(predictions, references, **kwargs)
    else:
        raise ValueError(f"Unknown score type: {score_type}")

# def qa_f1_zh_score(predictions: list[str], references: list[str], **kwargs) -> float:
#     prediction, ground_truth = predictions[0], references[0]
#     prediction_tokens = list(jieba.cut(prediction, cut_all=False))
#     ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
#     prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
#     ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
#     prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
#     ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
#     return f1_score([prediction_tokens], [ground_truth_tokens])
