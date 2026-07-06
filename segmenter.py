from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from boundary_model import BoundaryClassifier, load_segmented_corpus

try:
    import jieba
except ImportError:  # pragma: no cover - optional dependency for comparison experiments
    jieba = None


class ChineseSegmenter:
    def __init__(self, dictionary: Iterable[str] | Dict[str, int]) -> None:
        if isinstance(dictionary, dict):
            word_freq = {word.strip(): max(1, int(freq)) for word, freq in dictionary.items() if word.strip()}
        else:
            word_freq = {word.strip(): 1 for word in dictionary if word.strip()}

        words = set(word_freq)
        if not words:
            raise ValueError("dictionary cannot be empty")
        self.dictionary: Set[str] = words
        self.word_freq: Dict[str, int] = word_freq
        self.max_len = max(len(word) for word in words)
        self.total_freq = sum(word_freq.values()) or len(words)

    def forward_max_match(self, text: str) -> List[str]:
        result: List[str] = []
        index = 0
        while index < len(text):
            piece = self._match_longest(text, index, forward=True)
            result.append(piece)
            index += len(piece)
        return result

    def backward_max_match(self, text: str) -> List[str]:
        result: List[str] = []
        index = len(text)
        while index > 0:
            piece = self._match_longest(text, index, forward=False)
            result.append(piece)
            index -= len(piece)
        result.reverse()
        return result

    def bidirectional_max_match(self, text: str) -> List[str]:
        forward = self.forward_max_match(text)
        backward = self.backward_max_match(text)

        if len(forward) != len(backward):
            return forward if len(forward) < len(backward) else backward

        forward_single = sum(1 for word in forward if len(word) == 1)
        backward_single = sum(1 for word in backward if len(word) == 1)
        if forward_single != backward_single:
            return forward if forward_single < backward_single else backward

        return forward

    def forward_min_match(self, text: str) -> List[str]:
        result: List[str] = []
        index = 0
        while index < len(text):
            piece = self._match_shortest(text, index, forward=True)
            result.append(piece)
            index += len(piece)
        return result

    def backward_min_match(self, text: str) -> List[str]:
        result: List[str] = []
        index = len(text)
        while index > 0:
            piece = self._match_shortest(text, index, forward=False)
            result.append(piece)
            index -= len(piece)
        result.reverse()
        return result

    def neighbor_match(self, text: str) -> List[str]:
        result: List[str] = []
        index = 0
        while index < len(text):
            if index + 2 > len(text):
                result.append(text[index:])
                break

            prefix = text[index : index + 2]
            best = prefix
            best_len = len(best)
            for size in range(2, min(self.max_len, len(text) - index) + 1):
                piece = text[index : index + size]
                if piece.startswith(prefix) and piece in self.dictionary and size > best_len:
                    best = piece
                    best_len = size

            if prefix not in self.dictionary and best == prefix:
                best = text[index]

            result.append(best)
            index += len(best)
        return result

    def shortest_path_match(self, text: str) -> List[str]:
        if len(text) < 2:
            return [text] if text else []

        n = len(text)
        dist = [math.inf] * (n + 1)
        prev = [-1] * (n + 1)
        dist[0] = 0.0
        log_total = math.log(max(1, self.total_freq))

        for start in range(n):
            if math.isinf(dist[start]):
                continue
            max_end = min(n, start + self.max_len)
            for end in range(start + 1, max_end + 1):
                word = text[start:end]
                freq = self.word_freq.get(word)
                if freq is None and len(word) != 1:
                    continue
                weight = log_total - math.log(freq or 1)
                if dist[start] + weight < dist[end]:
                    dist[end] = dist[start] + weight
                    prev[end] = start

        if prev[n] == -1:
            return self.bidirectional_max_match(text)

        words: List[str] = []
        cursor = n
        while cursor > 0:
            start = prev[cursor]
            if start < 0:
                return self.bidirectional_max_match(text)
            words.append(text[start:cursor])
            cursor = start
        words.reverse()
        return words

    def hybrid_segment(self, text: str) -> List[str]:
        base_result = self.bidirectional_max_match(text)
        if jieba is None:
            return base_result

        refined: List[str] = []
        buffer: List[str] = []

        def flush_buffer() -> None:
            if not buffer:
                return
            chunk = "".join(buffer)
            if len(chunk) == 1:
                refined.append(chunk)
            else:
                jieba_words = [word for word in jieba.lcut(chunk) if word.strip()]
                if len(jieba_words) == 1 and len(jieba_words[0]) == len(chunk):
                    refined.extend(list(chunk))
                else:
                    refined.extend(jieba_words)
            buffer.clear()

        for word in base_result:
            if len(word) == 1:
                buffer.append(word)
                continue
            flush_buffer()
            refined.append(word)

        flush_buffer()
        return refined

    def rule_ml_segment(self, text: str, classifier: BoundaryClassifier | None) -> List[str]:
        if classifier is None:
            return self.bidirectional_max_match(text)
        return classifier.predict(text, self)

    def _match_longest(self, text: str, index: int, forward: bool) -> str:
        limit = min(self.max_len, len(text) - index) if forward else min(self.max_len, index)
        for size in range(limit, 0, -1):
            if forward:
                piece = text[index : index + size]
            else:
                piece = text[index - size : index]
            if piece in self.dictionary:
                return piece

        return text[index] if forward else text[index - 1]

    def _match_shortest(self, text: str, index: int, forward: bool) -> str:
        limit = min(self.max_len, len(text) - index) if forward else min(self.max_len, index)
        if limit < 2:
            return text[index] if forward else text[index - 1]

        for size in range(2, limit + 1):
            if forward:
                piece = text[index : index + size]
            else:
                piece = text[index - size : index]
            if piece in self.dictionary:
                return piece

        return text[index] if forward else text[index - 1]


def load_dictionary(path: Path) -> Dict[str, int]:
    entries: Dict[str, int] = {}
    dictionary_paths = [path]
    extra_path = path.with_name("dictionary_extra.txt")
    if extra_path.exists():
        dictionary_paths.append(extra_path)

    for dictionary_path in dictionary_paths:
        for line in dictionary_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 2 and parts[-1].isdigit():
                word = "".join(parts[:-1])
                freq = int(parts[-1])
            else:
                word = stripped
                freq = 1
            entries[word] = max(entries.get(word, 0), freq)
    return entries


def load_sentences(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_single_char_ratio(words: Sequence[str]) -> float:
    if not words:
        return 0.0
    singles = sum(1 for word in words if len(word) == 1)
    return singles / len(words)


def summarize(words: Sequence[str]) -> str:
    ratio = evaluate_single_char_ratio(words)
    counts = Counter(words)
    top_items = ", ".join(f"{word}:{count}" for word, count in counts.most_common(5))
    return (
        f"分词结果: {' / '.join(words)}\n"
        f"词数: {len(words)}\n"
        f"单字词比例: {ratio:.2%}\n"
        f"高频词: {top_items if top_items else '无'}"
    )


def build_report(segmenter: ChineseSegmenter, text: str, classifier: BoundaryClassifier | None = None) -> str:
    forward = segmenter.forward_max_match(text)
    backward = segmenter.backward_max_match(text)
    forward_min = segmenter.forward_min_match(text)
    backward_min = segmenter.backward_min_match(text)
    final = segmenter.bidirectional_max_match(text)
    neighbor = segmenter.neighbor_match(text)
    shortest = segmenter.shortest_path_match(text)
    hybrid = segmenter.hybrid_segment(text)
    ml_result = segmenter.rule_ml_segment(text, classifier)
    return "\n".join(
        [
            f"原句: {text}",
            f"正向最大匹配: {' / '.join(forward)}",
            f"逆向最大匹配: {' / '.join(backward)}",
            f"双向最大匹配: {' / '.join(final)}",
            f"混合分词结果: {' / '.join(hybrid)}",
            f"规则+机器学习分词: {' / '.join(ml_result)}",
            summarize(ml_result),
        ]
    )


def build_report(segmenter: ChineseSegmenter, text: str, classifier: BoundaryClassifier | None = None) -> str:
    results = [
        ("FMM", segmenter.forward_max_match(text)),
        ("MINM", segmenter.forward_min_match(text)),
        ("BMM", segmenter.backward_max_match(text)),
        ("RMINM", segmenter.backward_min_match(text)),
        ("BM", segmenter.bidirectional_max_match(text)),
        ("NM", segmenter.neighbor_match(text)),
        ("SPM", segmenter.shortest_path_match(text)),
        ("Jieba hybrid", segmenter.hybrid_segment(text)),
        ("Rule + ML", segmenter.rule_ml_segment(text, classifier)),
    ]
    lines = [f"Text: {text}"]
    lines.extend(f"{name}: {' / '.join(words)}" for name, words in results)
    lines.append(summarize(results[-1][1]))
    return "\n".join(lines)


def build_comparison_report(
    segmenter: ChineseSegmenter,
    text: str,
    classifier: BoundaryClassifier | None = None,
) -> str:
    manual_result = segmenter.bidirectional_max_match(text)
    hybrid_result = segmenter.hybrid_segment(text)
    ml_result = segmenter.rule_ml_segment(text, classifier)
    lines = [
        f"原句: {text}",
        f"双向最大匹配分词: {' / '.join(manual_result)}",
        f"混合分词: {' / '.join(hybrid_result)}",
        f"规则+机器学习分词: {' / '.join(ml_result)}",
    ]
    if jieba is None:
        lines.append("jieba 分词: 当前环境未安装 jieba，无法生成对比结果")
    else:
        jieba_result = jieba.lcut(text)
        lines.append(f"jieba 分词: {' / '.join(jieba_result)}")
    lines.extend(
        [
            f"双向最大匹配词数: {len(manual_result)}",
            f"双向最大匹配单字词比例: {evaluate_single_char_ratio(manual_result):.2%}",
            f"混合分词词数: {len(hybrid_result)}",
            f"混合分词单字词比例: {evaluate_single_char_ratio(hybrid_result):.2%}",
            f"规则+机器学习词数: {len(ml_result)}",
            f"规则+机器学习单字词比例: {evaluate_single_char_ratio(ml_result):.2%}",
        ]
    )
    return "\n".join(lines)


def process_file(
    segmenter: ChineseSegmenter,
    input_path: Path,
    output_path: Path,
    classifier: BoundaryClassifier | None = None,
) -> None:
    sentences = load_sentences(input_path)
    reports = []
    for sentence in sentences:
        reports.append(build_report(segmenter, sentence, classifier))
        reports.append("-" * 40)

    if reports:
        reports.pop()

    output_path.write_text("\n".join(reports), encoding="utf-8")


def process_comparison_file(
    segmenter: ChineseSegmenter,
    input_path: Path,
    output_path: Path,
    classifier: BoundaryClassifier | None = None,
) -> None:
    sentences = load_sentences(input_path)
    reports = []
    for sentence in sentences:
        reports.append(build_comparison_report(segmenter, sentence, classifier))
        reports.append("-" * 40)

    if reports:
        reports.pop()

    output_path.write_text("\n".join(reports), encoding="utf-8")


def interactive_mode(segmenter: ChineseSegmenter, classifier: BoundaryClassifier | None = None) -> None:
    samples = [
        "我爱自然语言处理",
        "南京市长江大桥",
        "研究生命起源",
        "他来到了网易杭研大厦",
    ]

    print("中文分词程序")
    print("1. 输入自己的句子")
    print("2. 运行内置示例")
    print("3. 从文件读取并输出到文件")
    print("4. 查看混合分词与 jieba 对比")
    print("5. 训练规则+机器学习模型")
    choice = input("请选择功能(1/2/3/4/5): ").strip()

    if choice == "1":
        text = input("请输入一句中文: ").strip()
        if not text:
            print("输入为空，程序结束。")
            return
        print(build_report(segmenter, text, classifier))
        return

    if choice == "3":
        input_name = input("请输入输入文件名: ").strip() or "input.txt"
        output_name = input("请输入输出文件名: ").strip() or "output.txt"
        process_file(segmenter, Path(input_name), Path(output_name), classifier)
        print(f"处理完成，结果已写入 {output_name}")
        return

    if choice == "4":
        text = input("请输入一句中文用于对比: ").strip()
        if not text:
            print("输入为空，程序结束。")
            return
        print(build_comparison_report(segmenter, text, classifier))
        return

    if choice == "5":
        print("请使用命令行参数 --train-ml 训练模型。")
        return

    for text in samples:
        print("=" * 40)
        print(build_report(segmenter, text, classifier))


def train_rule_ml_model(
    segmenter: ChineseSegmenter,
    corpus_path: Path,
    model_path: Path,
) -> BoundaryClassifier:
    sentences = load_segmented_corpus(corpus_path)
    classifier = BoundaryClassifier(dictionary=segmenter.dictionary, max_len=segmenter.max_len)
    classifier.train(sentences, segmenter)
    classifier.save(model_path)
    return classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于最大匹配与 jieba 回退的中文混合分词程序")
    parser.add_argument(
        "-d",
        "--dictionary",
        default="dictionary.txt",
        help="词典文件路径，默认使用当前目录下的 dictionary.txt",
    )
    parser.add_argument("-i", "--input", help="输入文本文件路径，每行一条待分词句子")
    parser.add_argument("-o", "--output", help="输出结果文件路径")
    parser.add_argument("-t", "--text", help="直接输入一条待分词句子")
    parser.add_argument("--train-ml", action="store_true", help="使用人工标注语料训练规则+机器学习边界模型")
    parser.add_argument("--train-corpus", default="training_corpus.txt", help="训练语料路径，格式为已分词句子")
    parser.add_argument("--model-path", default="boundary_model.pkl", help="规则+机器学习模型保存路径")
    parser.add_argument("--compare-text", help="输入一条句子，输出双向最大匹配、混合分词与 jieba 的对比结果")
    parser.add_argument("--compare-input", help="输入文本文件路径，批量生成双向最大匹配、混合分词与 jieba 的对比结果")
    parser.add_argument("--compare-output", help="对比结果输出文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    dictionary_path = Path(args.dictionary)
    if not dictionary_path.is_absolute():
        dictionary_path = base_dir / dictionary_path

    dictionary = load_dictionary(dictionary_path)
    segmenter = ChineseSegmenter(dictionary)
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = base_dir / model_path

    classifier = BoundaryClassifier.load(model_path) if model_path.exists() else None

    if args.train_ml:
        corpus_path = Path(args.train_corpus)
        if not corpus_path.is_absolute():
            corpus_path = base_dir / corpus_path
        classifier = train_rule_ml_model(segmenter, corpus_path, model_path)
        print(f"规则+机器学习模型训练完成，已写入 {model_path}")
        return

    if args.text:
        print(build_report(segmenter, args.text.strip(), classifier))
        return

    if args.compare_text:
        print(build_comparison_report(segmenter, args.compare_text.strip(), classifier))
        return

    if args.input and args.output:
        input_path = Path(args.input)
        output_path = Path(args.output)
        if not input_path.is_absolute():
            input_path = base_dir / input_path
        if not output_path.is_absolute():
            output_path = base_dir / output_path
        process_file(segmenter, input_path, output_path, classifier)
        print(f"处理完成，结果已写入 {output_path}")
        return

    if args.compare_input and args.compare_output:
        input_path = Path(args.compare_input)
        output_path = Path(args.compare_output)
        if not input_path.is_absolute():
            input_path = base_dir / input_path
        if not output_path.is_absolute():
            output_path = base_dir / output_path
        process_comparison_file(segmenter, input_path, output_path, classifier)
        print(f"对比完成，结果已写入 {output_path}")
        return

    interactive_mode(segmenter, classifier)


if __name__ == "__main__":
    main()
