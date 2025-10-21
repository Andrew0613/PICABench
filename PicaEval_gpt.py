#!/usr/bin/env python3

"""GPT-based PhysEdit evaluation (v2).
Matches the Qwen evaluator input/output contract while delegating inference to OpenAI GPT."""

import os
import sys
import json
import argparse
import random
import time
import base64
import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw
from tqdm import tqdm
from openai import OpenAI

# ---------------------------------------------------------------------------
# Shared data structures & helpers (kept aligned with PicaEval_qwen)
# ---------------------------------------------------------------------------

IMAGE_PATH_KEYS: Tuple[str, ...] = ("output_path", "output_image_path", "output_img_path")
JSON_PATTERN = re.compile(r"\{[^{}]*\"answer\"[^{}]*\"explanation\"[^{}]*\}", re.IGNORECASE | re.DOTALL)


@dataclass
class QATask:
    item_index: int
    qa_field: str
    qa_type: str
    qa_index: int
    image_path: str
    question: str
    answer: str
    source: str


def resolve_image_path(item: Dict[str, Any], base_dir: str) -> Optional[str]:
    for key in IMAGE_PATH_KEYS:
        rel_path = item.get(key)
        if rel_path:
            return os.path.join(base_dir, rel_path)
    return None


def iter_qa_entries(container: Any, default_type: str) -> List[Tuple[str, int, Dict[str, Any]]]:
    entries: List[Tuple[str, int, Dict[str, Any]]] = []
    if isinstance(container, dict):
        for qa_type, qa_list in container.items():
            for idx, qa in enumerate(qa_list):
                entries.append((qa_type, idx, qa))
    elif isinstance(container, list):
        for idx, qa in enumerate(container):
            entries.append((default_type, idx, qa))
    return entries


def draw_box_on_image(image: Image.Image, box_info: Dict[str, Any], box_color: str = "red") -> Image.Image:
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    x = box_info.get("x", 0)
    y = box_info.get("y", 0)
    width = box_info.get("width", 0)
    height = box_info.get("height", 0)
    bbox = [x, y, x + width, y + height]
    draw.rectangle(bbox, outline=box_color, width=3)
    return img_copy


def resize_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    long_edge = max(width, height)
    if long_edge == 1024:
        return image
    scale = 1024.0 / long_edge
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height), Image.LANCZOS)


def crop_image_with_box(image: Image.Image, box_info: Dict[str, Any], padding: int = 20) -> Image.Image:
    x = box_info.get("x", 0)
    y = box_info.get("y", 0)
    width = box_info.get("width", 0)
    height = box_info.get("height", 0)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.size[0], x + width + padding)
    y2 = min(image.size[1], y + height + padding)
    return image.crop((x1, y1, x2, y2))


def generate_viz_filename(item_index: int, qa_type: str, question_index: int, viz_mode: str) -> str:
    qa_type_short = qa_type.replace(" ", "_").replace("QA", "qa")
    return f"{item_index:04d}_{qa_type_short}_{question_index:03d}_{viz_mode}.jpg"


def create_visualization_and_question(
    output_img_path: str,
    qa_info: Dict[str, Any],
    item_index: int,
    qa_type: str,
    qa_index: int,
    viz_mode: str,
    viz_dir: str,
    args,
) -> Tuple[str, str]:
    question = qa_info.get("question", "")
    box_info = qa_info.get("box", {})
    viz_filename = generate_viz_filename(item_index, qa_type, qa_index, viz_mode)
    viz_path = os.path.join(viz_dir, viz_filename)
    viz_rel_path = os.path.join(f"visualization_annotated_qa_{viz_mode}", viz_filename)
    os.makedirs(viz_dir, exist_ok=True)
    try:
        image = Image.open(output_img_path).convert("RGB")
        image_resized = resize_image(image)
        if viz_mode == "draw_box":
            gpt_question = question
            viz_image = draw_box_on_image(image_resized, box_info, args.box_color)
        elif viz_mode == "crop_box":
            gpt_question = question
            viz_image = crop_image_with_box(image_resized, box_info, args.viz_padding)
        elif viz_mode == "crop_box_and_resize":
            gpt_question = question
            viz_image = crop_image_with_box(image_resized, box_info, args.viz_padding)
            viz_image = resize_image(viz_image)
        else:
            raise ValueError(f"Unknown viz_mode: {viz_mode}")
        viz_image.save(viz_path, quality=90)
        return gpt_question, viz_rel_path
    except Exception as exc:
        print(f"Error creating visualization for {viz_path}: {exc}")
        return question, ""


def _normalize_answer(answer: str) -> str:
    answer_lower = answer.lower().strip().rstrip('.')
    if answer_lower in ["yes", "y", "true"]:
        return "Yes"
    if answer_lower in ["no", "n", "false"]:
        return "No"
    return answer.strip()


def _parse_json_response(response: str) -> Optional[Tuple[str, str]]:
    try:
        response_clean = response.strip()
        json_candidates: List[str] = []
        json_start = response_clean.find('{')
        json_end = response_clean.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_candidates.append(response_clean[json_start:json_end])
        json_candidates.extend(JSON_PATTERN.findall(response_clean))
        for json_str in json_candidates:
            try:
                data = json.loads(json_str)
                answer = data.get("answer", "").strip()
                explanation = data.get("explanation", "").strip()
                if answer:
                    return _normalize_answer(answer), explanation
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    return None


def extract_yes_no_answer(response: str) -> str:
    response_lower = response.lower().strip()
    if response_lower.startswith("yes"):
        return "Yes"
    if response_lower.startswith("no"):
        return "No"
    if re.search(r"\byes\b", response_lower):
        return "Yes"
    if re.search(r"\bno\b", response_lower):
        return "No"
    return response[:10] if response else "Unknown"


def parse_structured_response(response: str) -> Tuple[str, str]:
    parsed = _parse_json_response(response)
    if parsed:
        return parsed
    return extract_yes_no_answer(response), ""


def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_image_base64(image_path: str, cache: Dict[str, str]) -> str:
    if image_path in cache:
        return cache[image_path]
    try:
        image = Image.open(image_path).convert("RGB")
        image = resize_image(image)
    except Exception as exc:
        print(f"Error loading image {image_path}: {exc}")
        image = Image.new("RGB", (512, 512), "white")
    encoded = encode_image(image)
    cache[image_path] = encoded
    return encoded


def create_structured_prompt(question: str) -> str:
    json_instruction = (
        "\n\nPlease provide a structured answer in the following JSON format:\n"
        '{"answer": "Yes" or "No", "explanation": "Brief explanation of your reasoning"}\n\n'
        "Output ONLY valid JSON. No extra text."
    )
    return question + json_instruction


def call_gpt_with_retries(client: OpenAI, prompt: str, image_data_url: str, args) -> str:
    for attempt in range(args.max_attempts):
        try:
            input_payload = [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }]
            response = client.responses.create(
                model=args.gpt_model,
                input=input_payload,
                reasoning={"effort": args.reasoning_effort},
            )
            return response.output_text.strip()
        except Exception as exc:
            wait_time = (args.retry_backoff ** attempt) + random.uniform(0, 1)
            print(f"GPT call failed ({attempt + 1}/{args.max_attempts}): {exc}; retry in {wait_time:.1f}s")
            time.sleep(wait_time)
    return "Error: all attempts failed"


def get_qa_entry(item: Dict[str, Any], qa_field: str, qa_type: str, qa_index: int) -> Dict[str, Any]:
    container = item.get(qa_field)
    if container is None and qa_field == "qa_pairs":
        container = item.get("annotated_qa_pairs")
    if isinstance(container, dict):
        return container[qa_type][qa_index]
    if isinstance(container, list):
        return container[qa_index]
    raise KeyError(f"QA entry not found for field={qa_field}, type={qa_type}, index={qa_index}")


def extract_qa_tasks_standard(items: List[Dict[str, Any]], image_base_dir: str) -> List[QATask]:
    qa_tasks: List[QATask] = []
    for item_idx, item in enumerate(items):
        image_path = resolve_image_path(item, image_base_dir)
        if not image_path:
            continue
        qa_container = item.get("qa_pairs")
        if qa_container is None:
            qa_container = item.get("annotated_qa_pairs", {})
        for qa_type, qa_idx, qa in iter_qa_entries(qa_container, "qa"):
            question = qa.get("question")
            answer = qa.get("answer")
            if question and answer:
                qa_tasks.append(QATask(item_idx, "qa_pairs", qa_type, qa_idx, image_path, question, answer, "original"))
    return qa_tasks


def extract_qa_tasks_annotated(items: List[Dict[str, Any]], image_base_dir: str, viz_mode: str, args) -> List[QATask]:
    qa_tasks: List[QATask] = []
    viz_dir = os.path.join(image_base_dir, f"visualization_annotated_qa_{viz_mode}")
    for item_idx, item in enumerate(items):
        image_path = resolve_image_path(item, image_base_dir)
        if not image_path:
            continue
        qa_container = item.get("annotated_qa_pairs", {})
        for qa_type, qa_idx, qa in iter_qa_entries(qa_container, "annotated_qa"):
            question = qa.get("question")
            answer = qa.get("answer")
            if not (question and answer):
                continue
            gpt_question, viz_rel_path = create_visualization_and_question(image_path, qa, item_idx, qa_type, qa_idx, viz_mode, viz_dir, args)
            viz_path = os.path.join(image_base_dir, viz_rel_path) if viz_rel_path else image_path
            source = "visualization" if viz_rel_path else "original"
            qa_tasks.append(QATask(item_idx, "annotated_qa_pairs", qa_type, qa_idx, viz_path, gpt_question, answer, source))
    return qa_tasks


def evaluate_with_gpt(items: List[Dict[str, Any]], image_base_dir: str, args) -> List[Dict[str, Any]]:
    if args.max_num is not None:
        items = items[:args.max_num]
    if args.qa_field == "qa_pairs":
        qa_tasks = extract_qa_tasks_standard(items, image_base_dir)
    elif args.qa_field == "annotated_qa_pairs":
        qa_tasks = extract_qa_tasks_annotated(items, image_base_dir, args.viz_mode, args)
    else:
        raise ValueError(f"Unsupported qa_field: {args.qa_field}")
    if not qa_tasks:
        print("No QA tasks found!")
        return items
    print(f"Found {len(qa_tasks)} QA tasks")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key is required. Set --api_key or OPENAI_API_KEY.")
    client = OpenAI(api_key=api_key)
    base64_cache: Dict[str, str] = {}
    for task in tqdm(qa_tasks, desc="Calling GPT model"):
        base64_str = load_image_base64(task.image_path, base64_cache)
        data_url = f"data:image/jpeg;base64,{base64_str}"
        prompt = create_structured_prompt(task.question)
        model_response = call_gpt_with_retries(client, prompt, data_url, args)
        model_answer, model_explanation = parse_structured_response(model_response)
        gt_clean = task.answer.lower().strip().rstrip('.')
        model_clean = model_answer.lower().strip().rstrip('.')
        is_correct = gt_clean == model_clean
        qa_entry = get_qa_entry(items[task.item_index], task.qa_field, task.qa_type, task.qa_index)
        qa_entry["model_answer"] = model_answer
        qa_entry["model_response"] = model_response
        qa_entry["model_explanation"] = model_explanation
        qa_entry["is_correct"] = is_correct
        if task.qa_field == "annotated_qa_pairs" and task.source == "visualization":
            viz_rel_path = os.path.relpath(task.image_path, image_base_dir)
            qa_entry["visualization_path"] = viz_rel_path
            qa_entry["viz_mode"] = args.viz_mode
    return items


def calculate_accuracy_by_dimension(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_questions = 0
    sample_acc_sum = 0.0
    sample_count = 0
    category_stats: Dict[str, Dict[str, float]] = {}
    law_stats: Dict[str, Dict[str, float]] = {}
    operation_stats: Dict[str, Dict[str, float]] = {}

    def collect_qas(item: Dict[str, Any]) -> List[Dict[str, Any]]:
        qa_sources: List[Dict[str, Any]] = []
        for field, default_type in (("qa_pairs", "qa"), ("annotated_qa_pairs", "annotated_qa")):
            container = item.get(field)
            if not container:
                continue
            for _, _, qa in iter_qa_entries(container, default_type):
                qa_sources.append(qa)
        return qa_sources

    def update_stat(stats: Dict[str, Dict[str, float]], key: str, sample_acc: float, qa_total: int) -> None:
        if key not in stats:
            stats[key] = {"sum_acc": 0.0, "sample_count": 0, "qa_total": 0}
        stats[key]["sum_acc"] += sample_acc
        stats[key]["sample_count"] += 1
        stats[key]["qa_total"] += qa_total

    for item in items:
        category = item.get("physics_category", "unknown")
        law = item.get("physics_law", "unknown")
        operation = item.get("edit_operation", "unknown")
        qa_sources = collect_qas(item)
        sample_total = 0
        sample_correct = 0
        for qa in qa_sources:
            if "is_correct" in qa:
                sample_total += 1
                if qa["is_correct"]:
                    sample_correct += 1
        if sample_total == 0:
            continue
        sample_acc = sample_correct / sample_total
        sample_count += 1
        sample_acc_sum += sample_acc
        total_questions += sample_total
        update_stat(category_stats, category, sample_acc, sample_total)
        update_stat(law_stats, law, sample_acc, sample_total)
        update_stat(operation_stats, operation, sample_acc, sample_total)

    def calc_accuracy(stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in stats.items():
            acc = 100.0 * value["sum_acc"] / value["sample_count"] if value["sample_count"] > 0 else 0.0
            result[key] = {
                "accuracy": acc,
                "sample_count": value["sample_count"],
                "qa_total": value["qa_total"],
            }
        return result

    overall_accuracy = (100.0 * sample_acc_sum / sample_count) if sample_count > 0 else 0.0
    return {
        "overall_accuracy": overall_accuracy,
        "sample_count": sample_count,
        "qa_total": total_questions,
        "by_category": calc_accuracy(category_stats),
        "by_law": calc_accuracy(law_stats),
        "by_operation": calc_accuracy(operation_stats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", type=str, required=True, help="Path to meta_info.json")
    parser.add_argument("--image_base_dir", type=str, default=None, help="Image base directory; defaults to JSON directory")
    parser.add_argument("--qa_field", type=str, default="annotated_qa_pairs", choices=["qa_pairs", "annotated_qa_pairs"], help="Select qa field")
    parser.add_argument("--viz_mode", type=str, default="crop_box_and_resize", choices=["draw_box", "crop_box", "crop_box_and_resize"], help="Visualization mode")
    parser.add_argument("--max_num", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--viz_padding", type=int, default=20, help="Padding pixels for crop mode")
    parser.add_argument("--box_color", type=str, default="red", help="Bounding box color")
    parser.add_argument("--log_question_changes", action="store_true", help="Log question mutations")
    parser.add_argument("--img_size", type=int, default=1024, help="Used for output naming consistency")
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API key; overrides OPENAI_API_KEY env")
    parser.add_argument("--gpt_model", type=str, default="gpt-5", help="OpenAI multimodal model name")
    parser.add_argument("--max_attempts", type=int, default=5, help="Max call retries")
    parser.add_argument("--retry_backoff", type=float, default=2.0, help="Exponential backoff base")
    parser.add_argument("--reasoning_effort", type=str, default="low", choices=["low", "medium", "high"], help="OpenAI reasoning effort setting")
    args = parser.parse_args()

    if args.image_base_dir is None:
        args.image_base_dir = os.path.dirname(args.input_json_path)

    print(f"Loading data: {args.input_json_path}")
    with open(args.input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"QA field: {args.qa_field}")
    if args.qa_field == "annotated_qa_pairs":
        print(f"Visualization mode: {args.viz_mode}")

    print("Running GPT evaluation...")
    results = evaluate_with_gpt(data, args.image_base_dir, args)

    out_dir = os.path.dirname(args.input_json_path)
    base_name = os.path.splitext(os.path.basename(args.input_json_path))[0]

    suffix = f"_gpt_output_{args.img_size}"
    if args.qa_field == "annotated_qa_pairs":
        suffix += f"_{args.viz_mode}"
    out_path = os.path.join(out_dir, base_name + suffix + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Computing statistics...")
    stats = calculate_accuracy_by_dimension(results)
    analysis_suffix = suffix.replace("_output_", "_analysis_")
    analysis_path = os.path.join(out_dir, base_name + analysis_suffix + ".json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== GPT evaluation finished ===")
    print(f"Overall accuracy: {stats['overall_accuracy']:.2f}% (samples: {stats['sample_count']}, qa_total: {stats['qa_total']})")
    print("\nBy category:")
    for category, stat in stats["by_category"].items():
        print(f"  {category}: {stat['accuracy']:.2f}% (samples: {stat['sample_count']}, qa_total: {stat['qa_total']})")
    print("\nBy law:")
    for law, stat in stats["by_law"].items():
        print(f"  {law}: {stat['accuracy']:.2f}% (samples: {stat['sample_count']}, qa_total: {stat['qa_total']})")
    print("\nBy operation:")
    for operation, stat in stats["by_operation"].items():
        print(f"  {operation}: {stat['accuracy']:.2f}% (samples: {stat['sample_count']}, qa_total: {stat['qa_total']})")

    print("\nOutputs saved:")
    print(f"  Detailed results: {out_path}")
    print(f"  Analysis: {analysis_path}")


if __name__ == "__main__":
    random.seed(42)
    main()
