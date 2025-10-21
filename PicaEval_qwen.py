#!/usr/bin/env python3
# physedit_eval_vllm.py
# PhysEdit评估脚本，基于vLLM进行批量推理
# 支持qa_pairs和annotated_qa_pairs两种模式
# 支持可视化功能（draw_box和crop_box模式）

import multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

import os
import sys
import json
import argparse
import random
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Iterable
from PIL import Image, ImageDraw
import PIL
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoProcessor

IMAGE_PATH_KEYS: Tuple[str, ...] = ("output_path", "output_image_path", "output_img_path")
JSON_PATTERN = re.compile(r'\{[^{}]*"answer"[^{}]*"explanation"[^{}]*\}', re.IGNORECASE | re.DOTALL)


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
    """根据常见字段推导图片完整路径"""
    for key in IMAGE_PATH_KEYS:
        rel_path = item.get(key)
        if rel_path:
            return os.path.join(base_dir, rel_path)
    return None

def iter_qa_entries(container: Any, default_type: str) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    """统一遍历问答结构，兼容dict/list两种格式"""
    if isinstance(container, dict):
        for qa_type, qa_list in container.items():
            for idx, qa in enumerate(qa_list):
                yield qa_type, idx, qa
    elif isinstance(container, list):
        for idx, qa in enumerate(container):
            yield default_type, idx, qa

def draw_box_on_image(image: Image.Image, box_info: Dict, box_color="red") -> Image.Image:
    """在图片上绘制bounding box"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # 获取box坐标
    x = box_info.get("x", 0)
    y = box_info.get("y", 0)
    width = box_info.get("width", 0)
    height = box_info.get("height", 0)
    
    # 绘制矩形框
    bbox = [x, y, x + width, y + height]
    draw.rectangle(bbox, outline=box_color, width=3)
    
    return img_copy

def resize_image(image: Image.Image) -> Image.Image:
    """将图片等比例resize到长边为1024"""
    width, height = image.size
    long_edge = max(width, height)
    
    if long_edge == 1024:
        return image
    
    scale = 1024.0 / long_edge
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return image.resize((new_width, new_height), Image.LANCZOS)

def crop_image_with_box(image: Image.Image, box_info: Dict, padding=20) -> Image.Image:
    """根据box坐标裁剪图片"""
    x = box_info.get("x", 0)
    y = box_info.get("y", 0)
    width = box_info.get("width", 0)
    height = box_info.get("height", 0)
    
    # 添加padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.size[0], x + width + padding)
    y2 = min(image.size[1], y + height + padding)
    cropped_image = image.crop((x1, y1, x2, y2))
    return cropped_image

def generate_viz_filename(item_index: int, qa_type: str, question_index: int, viz_mode: str) -> str:
    """生成可视化文件名"""
    qa_type_short = qa_type.replace(" ", "_").replace("QA", "qa")
    return f"{item_index:04d}_{qa_type_short}_{question_index:03d}_{viz_mode}.jpg"

def create_visualization_and_question(output_img_path: str, qa_info: Dict, item_index: int, 
                                     qa_type: str, qa_index: int, viz_mode: str, 
                                     viz_dir: str, args) -> Tuple[str, str]:
    """创建可视化图片并返回处理后的问题文本"""
    question = qa_info.get("question", "")
    box_info = qa_info.get("box", {})
    
    # 生成文件名和路径
    viz_filename = generate_viz_filename(item_index, qa_type, qa_index, viz_mode)
    viz_path = os.path.join(viz_dir, viz_filename)
    # 相对路径需要与上面的viz_dir一致，否则后续加载会找不到文件而回退为白图
    viz_rel_path = os.path.join(f"visualization_annotated_qa_{viz_mode}", viz_filename)
    
    # 确保目录存在
    os.makedirs(viz_dir, exist_ok=True)
    
    try:
        # 加载图片并resize到长边1024（因为bbox坐标是在长边1024下标注的）
        image = Image.open(output_img_path).convert("RGB")
        image_resized = resize_image(image)
        
        if viz_mode == "draw_box":
            # 画框模式：保留原始提问，仅在图上画框
            vllm_question = question
            viz_image = draw_box_on_image(image_resized, box_info, args.box_color)
        elif viz_mode == "crop_box":
            # 裁剪模式：保留原始提问，裁剪图片
            vllm_question = question
            viz_image = crop_image_with_box(image_resized, box_info, args.viz_padding)
        elif viz_mode == "crop_box_and_resize":
            # 裁剪模式：保留原始提问，裁剪图片再拉伸
            vllm_question = question
            viz_image = crop_image_with_box(image_resized, box_info, args.viz_padding)
            viz_image = resize_image(viz_image)
        else:
            raise ValueError(f"Unknown viz_mode: {viz_mode}")
        
        # 保存可视化图片
        viz_image.save(viz_path, quality=90)
        
        # 记录问题修改情况
        if args.log_question_changes and question != vllm_question:
            print(f"Question modified ({viz_mode}): {question} -> {vllm_question}")
        
        return vllm_question, viz_rel_path
        
    except Exception as e:
        print(f"Error creating visualization for {viz_path}: {e}")
        return question, ""

def parse_structured_response(response: str) -> Tuple[str, str]:
    """解析结构化的JSON回答"""
    return _parse_json_response(response) or _fallback_parse_response(response)

def _parse_json_response(response: str) -> Tuple[str, str] | None:
    """尝试解析JSON格式的回答"""
    try:
        response_clean = response.strip()
        
        # 多种方式查找JSON
        json_candidates = []
        
        # 方法1: 查找完整的{}块
        json_start = response_clean.find('{')
        json_end = response_clean.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_candidates.append(response_clean[json_start:json_end])
        
        # 方法2: 使用预编译正则表达式匹配JSON
        json_candidates.extend(JSON_PATTERN.findall(response_clean))
        
        # 尝试解析每个候选JSON
        for json_str in json_candidates:
            try:
                data = json.loads(json_str)
                answer = data.get("answer", "").strip()
                explanation = data.get("explanation", "").strip()
                
                if answer:  # 只要有answer就认为解析成功
                    return _normalize_answer(answer), explanation
            except json.JSONDecodeError:
                continue
                
    except Exception:
        pass
    
    return None

def _normalize_answer(answer: str) -> str:
    """标准化答案格式"""
    answer_lower = answer.lower().strip().rstrip('.')
    if answer_lower in ["yes", "y", "true"]:
        return "Yes"
    elif answer_lower in ["no", "n", "false"]:
        return "No"
    return answer.strip()

def _fallback_parse_response(response: str) -> Tuple[str, str]:
    """备用解析方法"""
    return extract_yes_no_answer(response), ""

def extract_yes_no_answer(response: str) -> str:
    """从模型回复中提取Yes/No答案（备用方法）"""
    response_lower = response.lower().strip()
    
    # 优先匹配开头的Yes/No
    if response_lower.startswith("yes"):
        return "Yes"
    elif response_lower.startswith("no"):
        return "No"
    
    # 查找句子中的Yes/No
    if re.search(r'\byes\b', response_lower):
        return "Yes"
    elif re.search(r'\bno\b', response_lower):
        return "No"
    
    # 默认返回原始回复的前10个字符
    return response[:10] if response else "Unknown"

def init_llm(model_path: str, tp: int, dtype: str, gpu_util: float, max_len: int):
    """初始化vLLM引擎"""
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        dtype=dtype,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
        max_model_len=max_len,
    )
    return llm

def create_structured_message(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """为消息添加结构化回答指令"""
    structured_msg = msgs.copy()
    if not structured_msg or structured_msg[-1]["role"] != "user":
        return structured_msg
    
    # 结构化回答指令
    json_instruction = "\n\nPlease provide a structured answer in the following JSON format:\n{\"answer\": \"Yes\" or \"No\", \"explanation\": \"Brief explanation of your reasoning\"}\n\nOutput ONLY valid JSON. No extra text."
    
    original_content = structured_msg[-1]["content"]
    if isinstance(original_content, list):
        # 多模态内容
        structured_content = original_content + [{"type": "text", "text": json_instruction}]
    else:
        # 纯文本内容
        structured_content = original_content + json_instruction
    
    structured_msg[-1]["content"] = structured_content
    return structured_msg

def prepare_vllm_batch(qa_tasks: List[QATask]) -> Tuple[List[List[Dict[str, Any]]], List[List[Image.Image]]]:
    """准备vLLM批处理的消息和图片"""
    msgs_batch: List[List[Dict[str, Any]]] = []
    imgs_batch: List[List[Image.Image]] = []
    pil_cache: Dict[str, Image.Image] = {}
    
    for task in tqdm(qa_tasks, desc="准备图片"):
        if task.image_path not in pil_cache:
            try:
                img = Image.open(task.image_path).convert("RGB")
                img = resize_image(img)
                pil_cache[task.image_path] = img
            except Exception as e:
                print(f"Error loading image {task.image_path}: {e}")
                pil_cache[task.image_path] = Image.new("RGB", (512, 512), "white")
        img = pil_cache[task.image_path]
        msgs_batch.append([{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": task.question}]}])
        imgs_batch.append([img])
    
    return msgs_batch, imgs_batch

def build_vllm_inputs(processor: AutoProcessor, 
                      batch_msgs: List[List[Dict[str, Any]]], 
                      batch_images: List[List[Image.Image]]) -> List[Dict[str, Any]]:
    """构建vLLM输入"""
    # 添加结构化回答指令
    structured_msgs = [create_structured_message(msgs) for msgs in batch_msgs]
    
    # 生成prompts
    prompts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in structured_msgs
    ]
    
    # 构造输入
    return [
        {"prompt": text, "multi_modal_data": {"image": imgs}}
        for text, imgs in zip(prompts, batch_images)
    ]

def vllm_generate_batch(llm: LLM, processor: AutoProcessor, 
                        batch_msgs: List[List[Dict[str, Any]]], 
                        batch_images: List[List[Image.Image]], 
                        max_new_tokens: int) -> List[str]:
    """批量推理"""
    # 构建输入
    inputs = build_vllm_inputs(processor, batch_msgs, batch_images)
    
    # 设置采样参数
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=["��", "\n��", "📐\n"]
    )
    
    # 批量推理
    outputs = llm.generate(inputs, sp)
    return [o.outputs[0].text if o.outputs else "" for o in outputs]

def get_qa_entry(item: Dict[str, Any], qa_field: str, qa_type: str, qa_index: int) -> Dict[str, Any]:
    """定位需要更新的QA条目"""
    container = item.get(qa_field)
    if container is None and qa_field == "qa_pairs":
        container = item.get("annotated_qa_pairs")
    if isinstance(container, dict):
        return container[qa_type][qa_index]
    if isinstance(container, list):
        return container[qa_index]
    raise KeyError(f"QA entry not found for field={qa_field}, type={qa_type}, index={qa_index}")


def extract_qa_tasks_standard(items: List[Dict[str, Any]], image_base_dir: str) -> List[QATask]:
    """提取qa_pairs模式的问答任务"""
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
                qa_tasks.append(QATask(
                    item_index=item_idx,
                    qa_field="qa_pairs",
                    qa_type=qa_type,
                    qa_index=qa_idx,
                    image_path=image_path,
                    question=question,
                    answer=answer,
                    source="original"
                ))
    return qa_tasks

def extract_qa_tasks_annotated(items: List[Dict[str, Any]], image_base_dir: str, viz_mode: str, args) -> List[QATask]:
    """提取annotated_qa_pairs模式的问答任务并生成可视化"""
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
            vllm_question, viz_rel_path = create_visualization_and_question(
                image_path, qa, item_idx, qa_type, qa_idx, viz_mode, viz_dir, args
            )
            viz_path = os.path.join(image_base_dir, viz_rel_path) if viz_rel_path else image_path
            source = "visualization" if viz_rel_path else "original"
            qa_tasks.append(QATask(
                item_index=item_idx,
                qa_field="annotated_qa_pairs",
                qa_type=qa_type,
                qa_index=qa_idx,
                image_path=viz_path,
                question=vllm_question,
                answer=answer,
                source=source
            ))
    return qa_tasks

def evaluate_physedit_with_vllm(items: List[Dict[str, Any]], image_base_dir: str, 
                                processor: AutoProcessor, llm: LLM, args) -> List[Dict[str, Any]]:
    """主评估函数"""
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
    msgs_batch, imgs_batch = prepare_vllm_batch(qa_tasks)
    print("开始VLM推理...")
    answers = vllm_generate_batch(llm, processor, msgs_batch, imgs_batch, args.max_new_tokens)
    
    for task, model_response in zip(qa_tasks, answers):
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
            viz_rel_path = os.path.relpath(task.image_path, args.image_base_dir)
            qa_entry["visualization_path"] = viz_rel_path
            qa_entry["viz_mode"] = args.viz_mode
    
    return items

def calculate_accuracy_by_dimension(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算各维度准确率（按样本平均）"""
    total_questions = 0
    sample_acc_sum = 0.0
    sample_count = 0
    category_stats = {}
    law_stats = {}
    operation_stats = {}
    
    def collect_qas(item: Dict[str, Any]) -> List[Dict[str, Any]]:
        qa_sources: List[Dict[str, Any]] = []
        for field, default_type in (("qa_pairs", "qa"), ("annotated_qa_pairs", "annotated_qa")):
            container = item.get(field)
            if not container:
                continue
            for _, _, qa in iter_qa_entries(container, default_type):
                qa_sources.append(qa)
        return qa_sources
    
    def update_stat(stats: Dict, key: str, sample_acc: float, qa_total: int):
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
    
    def calc_accuracy(stats: Dict) -> Dict:
        result = {}
        for key, value in stats.items():
            if value["sample_count"] > 0:
                acc = 100.0 * value["sum_acc"] / value["sample_count"]
            else:
                acc = 0.0
            result[key] = {
                "accuracy": acc,
                "sample_count": value["sample_count"],
                "qa_total": value["qa_total"]
            }
        return result
    
    overall_accuracy = (100.0 * sample_acc_sum / sample_count) if sample_count > 0 else 0.0
    return {
        "overall_accuracy": overall_accuracy,
        "sample_count": sample_count,
        "qa_total": total_questions,
        "by_category": calc_accuracy(category_stats),
        "by_law": calc_accuracy(law_stats),
        "by_operation": calc_accuracy(operation_stats)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", type=str, required=True,
                       help="输入的meta_info.json文件路径")
    parser.add_argument("--image_base_dir", type=str, default=None,
                       help="图片基础目录，默认为输入JSON的目录")
    parser.add_argument("--model_path", type=str, default="pretrained/Qwen/Qwen2.5-VL-72B-Instruct",
                       help="模型路径")
    parser.add_argument("--qa_field", type=str, default="annotated_qa_pairs", 
                       choices=["qa_pairs", "annotated_qa_pairs"],
                       help="选择使用qa_pairs还是annotated_qa_pairs字段")
    parser.add_argument("--viz_mode", type=str, default="crop_box_and_resize", 
                       choices=["draw_box", "crop_box", "crop_box_and_resize"],
                       help="可视化模式（仅annotated_qa_pairs模式有效）")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                       help="张量并行大小")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16","float16"])
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=5120)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--img_size", type=int, default=1024, choices=[512, 1024])
    parser.add_argument("--max_num", type=int, default=None, help="最大处理条数")
    parser.add_argument("--viz_padding", type=int, default=20, help="裁剪模式的padding像素")
    parser.add_argument("--box_color", default="red", help="绘制框的颜色")
    parser.add_argument("--log_question_changes", action="store_true", 
                       help="记录问题文本的修改情况")
    args = parser.parse_args()
    
    # 设置默认image_base_dir
    if args.image_base_dir is None:
        args.image_base_dir = os.path.dirname(args.input_json_path)
    
    # 读取数据
    print(f"加载数据: {args.input_json_path}")
    with open(args.input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"模式: {args.qa_field}")
    if args.qa_field == "annotated_qa_pairs":
        print(f"可视化模式: {args.viz_mode}")
    
    # 初始化模型
    print("初始化模型...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = init_llm(args.model_path, args.tensor_parallel_size, args.dtype, 
                   args.gpu_mem_util, args.max_model_len)
    
    # 评估
    print("开始评估...")
    results = evaluate_physedit_with_vllm(data, args.image_base_dir, processor, llm, args)
    
    # 保存结果
    out_dir = os.path.dirname(args.input_json_path)
    base_name = os.path.splitext(os.path.basename(args.input_json_path))[0]
    
    # 输出文件名
    suffix = f"_vllm_output_{args.img_size}"
    if args.qa_field == "annotated_qa_pairs":
        suffix += f"_{args.viz_mode}"
    
    out_path = os.path.join(out_dir, base_name + suffix + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 计算并保存统计结果
    print("计算统计结果...")
    stats = calculate_accuracy_by_dimension(results)
    
    analysis_suffix = suffix.replace("_output_", "_analysis_")
    analysis_path = os.path.join(out_dir, base_name + analysis_suffix + ".json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print(f"\n=== 评估完成 ===")
    print(f"总体准确率: {stats['overall_accuracy']:.2f}% "
          f"(samples: {stats['sample_count']}, qa_total: {stats['qa_total']})")
    print(f"\n按类别统计:")
    for category, stat in stats["by_category"].items():
        print(f"  {category}: {stat['accuracy']:.2f}% "
              f"(samples: {stat['sample_count']}, qa_total: {stat['qa_total']})")
    print(f"\n按物理定律统计:")
    for law, stat in stats["by_law"].items():
        print(f"  {law}: {stat['accuracy']:.2f}% "
              f"(samples: {stat['sample_count']}, qa_total: {stat['qa_total']})")
    print(f"\n按操作类型统计:")
    for operation, stat in stats["by_operation"].items():
        print(f"  {operation}: {stat['accuracy']:.2f}% "
              f"(samples: {stat['sample_count']}, qa_total: {stat['qa_total']})")
    
    print(f"\n文件已保存:")
    print(f"  详细结果: {out_path}")
    print(f"  统计分析: {analysis_path}")

if __name__ == "__main__":
    random.seed(42)
    main()
