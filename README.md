# Obsidian Docs Assistant

COMP6708 Assignment 3 · Group 21

这个本地文档助手回答 Obsidian、Dataview 和 Tasks 的使用问题。在英文或中文界面里，可以限定检索的手册、比较同一模型有无参考资料时的回答，并从回答里的引用跳到带行号的官方 Markdown 原文。系统使用检索增强生成（RAG），无需微调模型。

## 演示材料

- [课堂 PowerPoint](submission/Assignment3-Group21.pptx) 包含五页英文 slides，每页的演讲者备注中有操作提示。
- [英文旁白演示视频](submission/Assignment3-Group21-demo.mp4) 录制了实际的本地问答、来源核对和资料不足时的回答。播放器可开启英文字幕或按章节跳转。

## 在已准备好的演示机上运行

启动本机 Ollama 服务，然后从仓库根目录运行以下命令。已经运行的服务可以直接复用。

```sh
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

打开 <http://127.0.0.1:8501>。录制使用了 Ollama 中的 `qwen3-coder:30b` 和 `bge-m3`。模型、数据库和固定版本的原文都在本机。查询及展开引用支持离线操作，打开 GitHub 网页时需要联网。

## 从干净的仓库安装

需要 Python 3.11、Git、带 FTS5 的 SQLite 和 [Ollama](https://ollama.com/download)。在一个终端运行 `ollama serve`，在仓库根目录的另一个终端执行以下命令。首次安装、获取模型和文档需要联网。

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
ollama pull bge-m3
ollama pull qwen3:4b
python prepare_docs.py
python ingest.py
python -m streamlit run app.py --server.address 127.0.0.1
```

示例配置使用较小的 `qwen3:4b`。要复现录屏中的生成模型，请先拉取 `qwen3-coder:30b` 并将 `.env` 的 `OLLAMA_MODEL` 改为该名称。两个回答模式始终使用相同的生成模型。修改生成模型无需重建索引，修改向量模型则需要重新运行 `python ingest.py`。模型与 Git 检出的数据、Chroma 索引和机器专用的 `.env` 均不纳入版本控制。

保留了原有的 Hugging Face 部署方式。安装 `requirements-huggingface.txt`，在 `.env` 中设置 `LLM_BACKEND=huggingface`、`EMBEDDING_BACKEND=huggingface`、`QWEN_MODEL_PATH` 和 `EMBEDDING_MODEL_PATH`，再入库并启动。已有 v2 Chroma 索引可搭配原向量模型打开。

## 五分钟现场操作

1. **0:00–0:25**　展示封面。问观众，把工作标记为暂停后，它还会出现在未完成列表中吗？随即切换到应用。
2. **0:25–2:10**　仅选 **Tasks**，将 **Maximum sources** 设为 **3**，选择 **Compare**。提交示例 “Which status types does the Tasks ‘not done’ filter include?”。点击回答中的引用，对照状态表，再打开 **Filters › Filters for Task Statuses › Status**。第 285 行列出 `TODO`、`IN_PROGRESS` 和 `ON_HOLD`。可展开 **Show surrounding lines**。
3. **2:10–3:35**　选择 **New conversation**，切到 **Docs**，仅选 **Dataview**。点击隐藏默认 File 列的示例问题。展示 `TABLE WITHOUT ID`、复制按钮和原文引用。借这个例子解释语义检索与关键词检索如何互补。
4. **3:35–4:15**　输入 “Which room is Group 21 presenting in tomorrow?”。观察系统如何在手册缺少教室信息时作答。
5. **4:15–4:45**　展示末页的两项验证结果和一次漏检，为现场操作留出余量。

## 实现与验证

`prepare_docs.py` 按 `config.py` 中的三个 Git 提交下载 Markdown。`ingest.py` 按标题切分、保留代码与行号，并用 BGE-M3 向量化。固定版本共得到 **302 篇文档、3,300 个片段**。新集合写入完整后才切换活动索引，同一快照重复执行会复用已建好的索引。可以用 `python ingest.py --dry-run` 先查看切分统计。运行中的应用持有当前快照，重建后重启 Streamlit 即可加载新版本。

`rag.py` 在选中手册中同时使用 Chroma 的余弦相似度检索和 SQLite FTS5 的 BM25 检索，通过倒数排名融合（RRF）合并两个排序。对同一小节内相邻的命中进行合并，再把编号段落交给本地生成模型。界面显示的排名对应检索次序。每个引用保存仓库版本、文件和行号，回答模式与对照模式使用相同的问题和生成设置。

```sh
python check_demo.py
```

这组检查为每道题预先指定手册、文件和目标行。在录制所用的索引上，六题中有五题的前三个结果包含目标行，返回的 **18 个片段全部与对应版本的原始行范围一致**。一个关于“按完成日期计算下次重复时间”的中文问题漏掉了目标行。这两项检查衡量检索命中和出处对齐。总体回答准确率需独立评估。使用 `--strict` 可让任何目标行漏检返回非零退出状态。

三个文档来源分别为 [Obsidian Help](https://github.com/obsidianmd/obsidian-help/tree/ced32050c0d9d1e1f8d3faa19a8ba8474d75a32d)、[Dataview](https://github.com/blacksmithgu/obsidian-dataview/tree/5ad0994ff384cbb797de382e7edff2388141b73a) 和 [Tasks](https://github.com/obsidian-tasks-group/obsidian-tasks/tree/58f9960cd1d40429b1e2708659f7e37f352614ce)。
