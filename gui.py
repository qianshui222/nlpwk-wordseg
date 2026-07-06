from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from boundary_model import BoundaryClassifier
from segmenter import ChineseSegmenter, load_dictionary


BASE_DIR = Path(__file__).resolve().parent


class SegmenterApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("中文分词系统")
        self.geometry("1100x720")
        self.minsize(900, 560)

        self.dictionary_path = BASE_DIR / "dictionary.txt"
        self.model_path = BASE_DIR / "boundary_model.pkl"
        self.segmenter = ChineseSegmenter(load_dictionary(self.dictionary_path))
        self.classifier = self._load_classifier()

        self.algorithms = {
            "FMM 正向最大匹配": self.segmenter.forward_max_match,
            "MINM 正向最小匹配": self.segmenter.forward_min_match,
            "BMM 逆向最大匹配": self.segmenter.backward_max_match,
            "RMINM 逆向最小匹配": self.segmenter.backward_min_match,
            "BM 双向最大匹配": self.segmenter.bidirectional_max_match,
            "NM 邻近匹配": self.segmenter.neighbor_match,
            "SPM 最短路径匹配": self.segmenter.shortest_path_match,
            "Jieba 混合分词": self.segmenter.hybrid_segment,
            "规则 + 机器学习": lambda text: self.segmenter.rule_ml_segment(text, self.classifier),
        }

        self._build_layout()

    def _load_classifier(self) -> BoundaryClassifier | None:
        if not self.model_path.exists():
            return None
        try:
            return BoundaryClassifier.load(self.model_path)
        except Exception:
            return None

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(5, weight=1)

        ttk.Button(toolbar, text="打开文本", command=self.open_text).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(toolbar, text="运行选中算法", command=self.run_selected).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(toolbar, text="运行全部算法", command=self.run_all).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(toolbar, text="保存结果", command=self.save_result).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(toolbar, text="清空", command=self.clear_all).grid(row=0, column=4)

        self.status_var = tk.StringVar(value=f"词典: {self.dictionary_path.name}")
        ttk.Label(toolbar, textvariable=self.status_var, anchor="e").grid(row=0, column=5, sticky="e")

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        left = ttk.Frame(main, padding=(0, 0, 8, 0))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        main.add(left, weight=2)

        ttk.Label(left, text="输入文本").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.input_text = tk.Text(left, wrap="word", undo=True, height=10)
        self.input_text.grid(row=1, column=0, sticky="nsew")
        self.input_text.insert("1.0", "他说的确实在理，从小学到中学他都是好学生。")

        selector = ttk.Frame(left)
        selector.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        selector.columnconfigure(1, weight=1)
        ttk.Label(selector, text="算法").grid(row=0, column=0, padx=(0, 8))
        self.algorithm_var = tk.StringVar(value="BM 双向最大匹配")
        self.algorithm_box = ttk.Combobox(
            selector,
            textvariable=self.algorithm_var,
            values=list(self.algorithms),
            state="readonly",
        )
        self.algorithm_box.grid(row=0, column=1, sticky="ew")

        right = ttk.Frame(main)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        main.add(right, weight=3)

        ttk.Label(right, text="分词结果").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.result_text = tk.Text(right, wrap="word", undo=True)
        self.result_text.grid(row=1, column=0, sticky="nsew")

        table_frame = ttk.Frame(right)
        table_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        table_frame.columnconfigure(0, weight=1)

        self.result_table = ttk.Treeview(
            table_frame,
            columns=("algorithm", "words", "time"),
            show="headings",
            height=7,
        )
        self.result_table.heading("algorithm", text="算法")
        self.result_table.heading("words", text="词数")
        self.result_table.heading("time", text="耗时(ms)")
        self.result_table.column("algorithm", width=180, anchor="w")
        self.result_table.column("words", width=80, anchor="center")
        self.result_table.column("time", width=90, anchor="center")
        self.result_table.grid(row=0, column=0, sticky="ew")

    def open_text(self) -> None:
        path = filedialog.askopenfilename(
            title="选择文本文件",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        content = Path(path).read_text(encoding="utf-8")
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", content)
        self.status_var.set(f"已打开: {Path(path).name}")

    def save_result(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存分词结果",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        Path(path).write_text(self.result_text.get("1.0", "end").strip() + "\n", encoding="utf-8")
        self.status_var.set(f"已保存: {Path(path).name}")

    def clear_all(self) -> None:
        self.input_text.delete("1.0", "end")
        self.result_text.delete("1.0", "end")
        for item in self.result_table.get_children():
            self.result_table.delete(item)
        self.status_var.set(f"词典: {self.dictionary_path.name}")

    def run_selected(self) -> None:
        name = self.algorithm_var.get()
        self._run_algorithms([name])

    def run_all(self) -> None:
        self._run_algorithms(list(self.algorithms))

    def _run_algorithms(self, names: list[str]) -> None:
        text = self.input_text.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("提示", "请输入需要分词的文本。")
            return

        self.result_text.delete("1.0", "end")
        for item in self.result_table.get_children():
            self.result_table.delete(item)

        blocks = []
        for name in names:
            func = self.algorithms[name]
            start = time.perf_counter()
            words = func(text)
            elapsed = (time.perf_counter() - start) * 1000
            blocks.append(f"{name}\n{' / '.join(words)}\n耗时: {elapsed:.3f} ms")
            self.result_table.insert("", "end", values=(name, len(words), f"{elapsed:.3f}"))

        self.result_text.insert("1.0", "\n\n".join(blocks))
        self.status_var.set(f"完成: {len(names)} 个算法")


def main() -> None:
    app = SegmenterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
