import os
import sys
from typing import List

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from transformer_encoder.sql_vae_codet5 import CodeT5Encoder

MODEL_DIR = os.path.join(PROJECT_ROOT, "codet5-small")
INPUT_FILE = os.path.join(PROJECT_ROOT, "output", "predicate.txt")
OUTPUT_SENT_EMB_FILE = os.path.join(PROJECT_ROOT, "output", "predicate_embeddings_codet5.npy")
ENCODER_STATE_FILE = os.path.join(MODEL_DIR, "encoder_state.pt")
ENCODE_BATCH_SIZE = 32  # 批量编码大小；只影响吞吐，不改变谓词顺序和向量语义。
USE_GPU = torch.cuda.is_available()


def encode_sentence(
    text: str,
    tokenizer,
    model: CodeT5Encoder,
    device: torch.device,
) -> np.ndarray:
    """兼容旧调用的单句 Encoder mean-pooling 接口。"""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
    )
    with torch.no_grad():
        _token_hidden, pooled = model.encode_encoder_only(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
    return pooled[0].detach().cpu().numpy().astype(np.float32)


def run_encoding():
    device = torch.device("cuda" if USE_GPU else "cpu")
    print(f"[信息] 使用设备: {device}")

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"未找到本地模型目录：{MODEL_DIR}，请检查路径。")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"未找到输入文件：{INPUT_FILE}，请检查路径或文件名。")

    print(f"[信息] 正在加载 CodeT5 Encoder-only 模型: {MODEL_DIR}")
    model = CodeT5Encoder(
        model_name_or_path=MODEL_DIR,
        device=device,
    )
    if os.path.exists(ENCODER_STATE_FILE):
        print(f"[信息] 加载 Encoder checkpoint: {ENCODER_STATE_FILE}")
        model.load_encoder_state_dict(
            torch.load(ENCODER_STATE_FILE, map_location=device)
        )
    else:
        print("[信息] 未发现 encoder_state.pt，本轮从原始 CodeT5-small 权重开始。")

    model.eval()
    tokenizer = model.tokenizer
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"[信息] 共读取到 {len(lines)} 行谓词。")

    all_sent_embeddings: List[np.ndarray] = []
    for start in range(0, len(lines), ENCODE_BATCH_SIZE):
        batch_lines = lines[start : start + ENCODE_BATCH_SIZE]
        encoding = tokenizer(
            batch_lines,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            _token_hidden, pooled = model.encode_encoder_only(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            )
        all_sent_embeddings.extend(
            row.detach().cpu().numpy().astype(np.float32)
            for row in pooled
        )

        if start == 0 and len(pooled) > 0:
            print(f"[调试] Encoder pooled 向量形状: {tuple(pooled.shape)}")

    if len(all_sent_embeddings) != len(lines):
        raise RuntimeError(
            f"编码结果数量 {len(all_sent_embeddings)} 与谓词行数 {len(lines)} 不一致。"
        )

    sent_embeddings_matrix = np.stack(all_sent_embeddings, axis=0).astype(np.float32)
    if sent_embeddings_matrix.shape[1] != 512:
        raise RuntimeError(
            f"CodeT5 Encoder hidden size 应为 512，实际为 {sent_embeddings_matrix.shape[1]}。"
        )

    os.makedirs(os.path.dirname(OUTPUT_SENT_EMB_FILE), exist_ok=True)
    np.save(OUTPUT_SENT_EMB_FILE, sent_embeddings_matrix)
    print(f"[信息] 已保存 Encoder pooled 向量: {os.path.abspath(OUTPUT_SENT_EMB_FILE)}")
    print(f"[信息] 输出形状: {sent_embeddings_matrix.shape}（N, 512）")
