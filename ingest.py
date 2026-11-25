"""文档入库：读取 Markdown -> 清洗 -> 按标题切块 -> Embedding -> 写入 ChromaDB。

用法：
    python ingest.py            # 完整入库（会重建集合）
    python ingest.py --dry-run  # 只做加载和切块，打印统计，不加载模型
"""

import argparse
import hashlib
import re
import subprocess
import uuid
from collections import Counter
from urllib.parse import quote

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

import config

FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
# 代码块和行内代码里的 [[...]] 等是在讲解语法，必须原样保留
FENCE = re.compile(r"^[\s>]*(`{3,}|~{3,})")
FENCE_PREFIX = re.compile(r"^[\s>]*")
INLINE_CODE = re.compile(r"(`[^`]+`)")
HTML_COMMENT = re.compile(r"<!--.*?-->")
RELATED_PAGES = re.compile(r'<span class="related-pages">.*?</span>')
EMBED = re.compile(r"!\[\[([^\]]+)\]\]")
WIKILINK = re.compile(r"\[\[([^\]|\\]*)(?:\\?\|([^\]]+))?\]\]")
MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
BLOCK_ID = re.compile(r"[ \t]+\^[\w-]+[ \t]*$")
HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
FILE_EXTENSION = re.compile(r"\.\w{2,5}(#.*)?$")


def _replace_embed(match: re.Match) -> str:
    target = match.group(1).split("|")[0]
    if FILE_EXTENSION.search(target):
        return ""  # 图片、图标
    return f"(see {target.replace('#', ' > ')})"  # 嵌入的其他笔记


def _replace_wikilink(match: re.Match) -> str:
    target, alias = match.group(1), match.group(2)
    if alias:
        return alias
    return target.lstrip("#").replace("#", " > ")


def _clean_prose(text: str) -> str:
    text = HTML_COMMENT.sub("", text)
    text = RELATED_PAGES.sub("", text)
    text = EMBED.sub(_replace_embed, text)
    text = WIKILINK.sub(_replace_wikilink, text)
    return MD_IMAGE.sub("", text)


def mark_code_lines(text: str):
    """逐行产出 (行, 是否在围栏代码块内)。

    支持嵌套围栏（```` 里包 ```）以及引用块里的围栏（> ```）。
    """
    fence = None
    for line in text.split("\n"):
        match = FENCE.match(line)
        if fence is None:
            if match:
                fence = match.group(1)
            yield line, match is not None
        else:
            yield line, True
            closes = (
                match
                and match.group(1)[0] == fence[0]
                and len(match.group(1)) >= len(fence)
                and FENCE_PREFIX.sub("", line).strip() == match.group(1)
            )
            if closes:
                fence = None


def clean_markdown_with_lines(text: str) -> tuple[str, list[int]]:
    """清洗 Markdown，同时保留每个清洗后行对应的原文件行号。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    frontmatter = FRONTMATTER.match(text)
    first_line = 1
    if frontmatter:
        first_line += frontmatter.group(0).count("\n")
        text = text[frontmatter.end() :]

    lines: list[str] = []
    line_numbers: list[int] = []
    for original_line, (line, in_code) in enumerate(
        mark_code_lines(text), start=first_line
    ):
        if not in_code:
            # split 带捕获组：偶数下标是正文，奇数下标是行内代码
            parts = INLINE_CODE.split(line)
            line = "".join(
                part if index % 2 else _clean_prose(part)
                for index, part in enumerate(parts)
            )
            line = BLOCK_ID.sub("", line)
            if line.lstrip().startswith("|"):  # 表格行：压缩对齐用的空格和横线
                line = re.sub(r" {2,}", " ", line)
                line = re.sub(r"-{4,}", "---", line)
        line = line.rstrip()
        # 与原来的 re.sub(r"\n{3,}", "\n\n", ...) 一致：连续空行只留一个。
        if not line and lines and not lines[-1]:
            continue
        lines.append(line)
        line_numbers.append(original_line)

    while lines and not lines[0]:
        lines.pop(0)
        line_numbers.pop(0)
    while lines and not lines[-1]:
        lines.pop()
        line_numbers.pop()
    return "\n".join(lines), line_numbers


def clean_markdown(text: str) -> str:
    """去掉 frontmatter 和 Obsidian 专有标记，代码部分保持不变。"""
    return clean_markdown_with_lines(text)[0]


def split_by_headings_with_lines(
    text: str, line_numbers: list[int]
) -> list[tuple[list[str], str, list[int]]]:
    """按标题切成小节，并保留每一正文行在原文件中的行号。"""
    sections: list[tuple[list[str], str, list[int]]] = []
    path: list[str | None] = [None, None, None]
    body: list[str] = []
    body_line_numbers: list[int] = []

    def flush() -> None:
        while body and not body[0]:
            body.pop(0)
            body_line_numbers.pop(0)
        while body and not body[-1]:
            body.pop()
            body_line_numbers.pop()
        content = "\n".join(body)
        if content:
            sections.append(
                ([h for h in path if h], content, body_line_numbers.copy())
            )
        body.clear()
        body_line_numbers.clear()

    marked_lines = list(mark_code_lines(text))
    if len(marked_lines) != len(line_numbers):
        raise ValueError("清洗后的文本行数与原文行号映射不一致")
    for (line, in_code), original_line in zip(marked_lines, line_numbers):
        match = None if in_code else HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[level - 1] = match.group(2).strip()
            for deeper in range(level, 3):
                path[deeper] = None
        else:
            body.append(line)
            body_line_numbers.append(original_line)
    flush()
    return sections


def split_by_headings(text: str) -> list[tuple[list[str], str]]:
    """兼容原接口；没有原文映射时使用当前文本的行号。"""
    line_numbers = list(range(1, len(text.split("\n")) + 1))
    return [
        (path, body)
        for path, body, _ in split_by_headings_with_lines(text, line_numbers)
    ]


def _chunk_line_range(
    body: str, body_line_numbers: list[int], start: int, content: str
) -> tuple[int, int]:
    """把分块器给出的字符范围换算成原始 Markdown 的真实行号。"""
    start_index = body.count("\n", 0, start)
    end_offset = max(start, start + len(content) - 1)
    end_index = body.count("\n", 0, end_offset)
    return body_line_numbers[start_index], body_line_numbers[end_index]


def load_chunks() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks: list[Document] = []

    for source in config.SOURCES:
        repo_dir = config.DATA_DIR / source["dir"]
        docs_dir = repo_dir / source["docs"]
        if not docs_dir.is_dir():
            raise SystemExit(f"找不到 {docs_dir}，请先把仓库 clone 到 data/ 目录")

        versioned = (repo_dir / ".git").exists()
        revision = source["branch"]
        if versioned:
            revision = subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
            ).strip()
            entries = subprocess.check_output([
                "git", "-C", str(repo_dir), "ls-tree", "-rz", "--name-only", revision, "--", source["docs"]
            ]).decode("utf-8").split("\0")
            paths = (repo_dir / entry for entry in entries if entry.endswith(".md"))
        else:
            paths = docs_dir.rglob("*.md")
        file_count = 0
        for path in sorted(paths):
            relative_to_docs = path.relative_to(docs_dir)
            if path.name in source["skip_files"]:
                continue
            if source["skip_dirs"] & set(relative_to_docs.parts):
                continue

            relative = path.relative_to(repo_dir).as_posix()
            if versioned:
                # Read the committed file even if the checkout contains local edits.
                raw_text = subprocess.check_output([
                    "git", "-C", str(repo_dir), "show", f"{revision}:{relative}"
                ]).decode("utf-8", errors="replace")
            else:
                raw_text = path.read_text(encoding="utf-8", errors="replace")
            text, line_numbers = clean_markdown_with_lines(raw_text)
            if len(text) < 50:
                continue
            file_count += 1

            url = (
                f"https://github.com/{source['repo']}/blob/"
                f"{revision}/{quote(relative)}"
            )
            # Obsidian Help 的页面没有一级标题，标题就是文件名
            first_heading = re.match(r"# +(.+)", text)
            title = first_heading.group(1).strip() if first_heading else path.stem

            doc_chunk_index = 0
            for section_index, (heading_path, body, body_line_numbers) in enumerate(
                split_by_headings_with_lines(text, line_numbers)
            ):
                trail = [title] + [h for h in heading_path if h != title]
                section = " > ".join(trail)
                section_key = f"{source['project']}|{relative}|{section_index}"
                section_id = hashlib.sha1(section_key.encode()).hexdigest()[:16]
                section_document = Document(page_content=body)
                for section_chunk_index, piece_document in enumerate(
                    splitter.split_documents([section_document])
                ):
                    piece = piece_document.page_content
                    start_index = piece_document.metadata["start_index"]
                    start_line, end_line = _chunk_line_range(
                        body, body_line_numbers, start_index, piece
                    )
                    chunk_key = f"{section_key}|{section_chunk_index}|{piece}"
                    chunk_id = hashlib.sha1(chunk_key.encode()).hexdigest()
                    chunks.append(
                        Document(
                            # 把出处写进文本，脱离上下文的小片段也能被正确检索
                            page_content=f"[{source['project']} docs: {section}]\n{piece}",
                            metadata={
                                "project": source["project"],
                                "title": title,
                                "section": section,
                                "source": relative,
                                "url": url,
                                "start_line": start_line,
                                "end_line": end_line,
                                "section_id": section_id,
                                "section_index": section_index,
                                "section_chunk_index": section_chunk_index,
                                "doc_chunk_index": doc_chunk_index,
                                "chunk_id": chunk_id,
                            },
                        )
                    )
                    doc_chunk_index += 1
        print(f"{source['project']:<9} {file_count:>4} 篇文档")

    return chunks


def print_stats(chunks: list[Document]) -> None:
    if not chunks:
        raise SystemExit("No Markdown passages found. Run python prepare_docs.py first.")
    per_project = Counter(chunk.metadata["project"] for chunk in chunks)
    lengths = sorted(len(chunk.page_content) for chunk in chunks)
    print(f"共 {len(chunks)} 个文本块：{dict(per_project)}")
    print(
        f"块长度（字符）：最短 {lengths[0]}，中位数 {lengths[len(lengths) // 2]}，"
        f"最长 {lengths[-1]}"
    )


def build_index(chunks: list[Document]) -> None:
    import chromadb
    from langchain_chroma import Chroma
    from rag import get_embeddings

    if not chunks:
        raise ValueError("Cannot build an empty index")
    fingerprint = hashlib.sha256(config.EMBEDDING_ID.encode())
    for chunk in chunks:
        for field in ("chunk_id", "url", "start_line", "end_line"):
            fingerprint.update(str(chunk.metadata[field]).encode() + b"\0")

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    existing = {collection.name for collection in client.list_collections()}
    active = config.active_collection()
    if active in existing:
        current = client.get_collection(active)
        metadata = current.metadata or {}
        if (metadata.get("fingerprint") == fingerprint.hexdigest()
                and metadata.get("schema_version") == config.INDEX_SCHEMA_VERSION
                and current.count() == len(chunks)):
            print(f"Already indexed: {len(chunks)} passages.")
            return

    print(f"Embedding with {config.EMBEDDING_ID}", flush=True)
    collection_name = f"{config.COLLECTION_NAME}-{uuid.uuid4().hex[:12]}"
    store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        collection_metadata={
            "hnsw:space": "cosine",
            # 查询时必须用同一个 Embedding 模型，这里记下来供 rag.py 校验
            "embedding_model": config.EMBEDDING_ID,
            "schema_version": config.INDEX_SCHEMA_VERSION,
            "fingerprint": fingerprint.hexdigest(),
            "documents": len({(d.metadata["project"], d.metadata["source"]) for d in chunks}),
        },
    )

    batch_size = 32
    try:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            ids = [chunk.metadata["chunk_id"] for chunk in batch]
            store.add_documents(batch, ids=ids)
            print(f"Indexed {start + len(batch)}/{len(chunks)}", flush=True)
        if client.get_collection(collection_name).count() != len(chunks):
            raise RuntimeError("Incomplete index")
        # Readers keep the previous collection until the complete replacement is published.
        new_pointer = config.ACTIVE_INDEX.with_name(f"{collection_name}.tmp")
        new_pointer.write_text(collection_name, encoding="utf-8")
        new_pointer.replace(config.ACTIVE_INDEX)
    except BaseException:
        client.delete_collection(collection_name)
        raise

    print(f"完成，向量库位于 {config.CHROMA_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只切块并打印统计")
    args = parser.parse_args()

    chunks = load_chunks()
    print_stats(chunks)
    if not args.dry_run:
        build_index(chunks)


if __name__ == "__main__":
    main()
