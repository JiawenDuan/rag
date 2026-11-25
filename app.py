"""Launch the documentation assistant with streamlit run app.py."""

import html
import re
import time
import uuid

import streamlit as st

import config
import rag

st.set_page_config(page_title="Obsidian Docs Assistant", page_icon="📖", layout="wide")
st.markdown("""
<style>
[data-testid="stMainBlockContainer"] {max-width: 1080px; padding-top: 3.6rem;}
[data-testid="stSidebar"] {border-right: 1px solid #e2dfea;}
[data-testid="stSidebar"] h2 {font-size: 1.05rem;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: .82rem;}
[data-testid="stSidebar"] hr {margin: .45rem 0;}
[data-testid="stHeader"] {background: transparent; height: 2.75rem;}
[data-testid="stDecoration"] {display: none;}
.intro {padding: 0 0 .8rem; border-bottom: 1px solid #ded9e7; margin-bottom: .35rem;}
.intro h1 {color: #262035; font-size: clamp(2rem, 4vw, 3.15rem); line-height: 1.08;
           padding: 0; margin: 0 0 .65rem; letter-spacing: -.05em;}
.intro p {color: #5a5465; line-height: 1.5; font-size: 1.05rem; margin: 0 0 .45rem;}
.answer-label {font-size: .81rem; font-weight: 700; color: #664a99; margin-bottom: .15rem;}
.source-anchor {scroll-margin-top: 3.8rem;}
.st-key-examples button {text-align: left; justify-content: flex-start; height: auto; min-height: 3.4rem;}
.st-key-examples button p {text-align: left; line-height: 1.45;}
.example-project {color: #684a98; font-size: .95rem; font-weight: 700;
                  border-top: 2px solid #b5a3cc; padding: .75rem 0 .1rem;}
.index-count {display: flex; justify-content: space-between; color: #595363; font-size: .87rem; line-height: 1.6;}
.doc-lines {border-left: 3px solid #8061ac; overflow-x: auto; max-height: 24rem;
            background: #f6f3fa; padding: .65rem .8rem; border-radius: .25rem;}
.doc-line {display: flex; gap: .85rem; font-family: ui-monospace, Menlo, monospace;
           font-size: .84rem; line-height: 1.6; color: #332741;}
.doc-line .ln {flex: 0 0 2rem; color: #736584; user-select: none; text-align: right;}
.doc-line .tx {white-space: pre-wrap; overflow-wrap: anywhere;}
.doc-line.context {opacity: .65;}
[data-testid="stChatMessage"] {border-radius: .6rem;}
</style>
""", unsafe_allow_html=True)

st.session_state.setdefault("turns", [])
with st.sidebar:
    language = st.selectbox("Interface language", ("English", "中文"), key="language")


def tr(en: str, zh: str) -> str:
    return en if language == "English" else zh


EXAMPLES = [
    ("Tasks", "Which status types does the Tasks ‘not done’ filter include?", "Tasks 的 not done 筛选器包含哪些状态类型？"),
    ("Dataview", "How do I hide the default File column in a Dataview table?", "如何隐藏 Dataview 表格中默认的 File 列？"),
    ("Obsidian", "How do I link to a heading in another note?", "如何链接到另一篇笔记里的某个标题？"),
    ("Tasks", "How do I query tasks due tomorrow?", "如何查询明天到期的任务？"),
    ("Dataview", "How do I sort a Dataview query in descending order?", "如何让 Dataview 查询按降序排序？"),
    ("Obsidian", "Which property types can a note have?", "笔记支持哪些属性类型？"),
]

CODE = re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]+`)", re.DOTALL)
CITATION = re.compile(r"\[(\d+)\](?!\()")


def anchor(turn_id: str, index: int) -> str:
    return f"evidence-{turn_id}-{index}"


def linked_answer(text: str, sources: list[rag.Source], turn_id: str) -> str:
    ids = {source.index for source in sources}

    def link(match: re.Match) -> str:
        number = int(match[1])
        return f"[\\[{number}\\]](#{anchor(turn_id, number)})" if number in ids else match[0]

    return "".join(part if i % 2 else CITATION.sub(link, part)
                   for i, part in enumerate(CODE.split(text)))


@st.cache_resource(show_spinner=False)
def load_index() -> dict:
    rag.get_vector_store()
    return rag.index_stats()


def line_rows(body: str, first: int, context: bool = False) -> str:
    if not body:
        return ""
    style = "doc-line context" if context else "doc-line"
    return "".join(
        f'<div class="{style}"><span class="ln">{n}</span><span class="tx">{html.escape(line)}</span></div>'
        for n, line in enumerate(body.split("\n"), first)
    )


def evidence(sources: list[rag.Source], turn_id: str) -> None:
    if not sources:
        return
    st.markdown(f"#### {tr('Source passages', '来源原文')}")
    for source in sources:
        section = source.section.replace(" > ", " › ")
        with st.expander(f"[{source.index}]  {source.project}  ·  {section}"):
            st.markdown(f'<span class="source-anchor" id="{anchor(turn_id, source.index)}"></span>',
                        unsafe_allow_html=True)
            st.caption(f"{source.source}  ·  {tr('lines', '行')} {source.start_line}–{source.end_line}")
            if source.exact_excerpt:
                context = st.toggle(tr("Show surrounding lines", "显示上下文"), key=f"context-{turn_id}-{source.index}")
                before = line_rows(source.context_before, source.context_start_line, True) if context else ""
                hit = line_rows(source.raw_content, source.start_line)
                after = line_rows(source.context_after, source.end_line + 1, True) if context else ""
                st.markdown(f'<div class="doc-lines">{before}{hit}{after}</div>', unsafe_allow_html=True)
            else:
                st.caption(tr("Indexed excerpt", "索引片段"))
                st.code(source.raw_content or source.content, language="markdown", wrap_lines=True)
            st.link_button(tr("Open source on GitHub", "在 GitHub 查看原文"), source.url)
            ranks = []
            if source.semantic_rank:
                ranks.append(f"{tr('Semantic rank', '语义排名')} {source.semantic_rank}")
            if source.keyword_rank:
                ranks.append(f"{tr('Keyword rank', '关键词排名')} {source.keyword_rank}")
            if ranks:
                st.caption("  ·  ".join(ranks))


def render_answer(turn: dict, kind: str, live: bool) -> None:
    grounded = kind == "rag"
    label = tr("With documentation", "基于文档") if grounded else tr("Model only", "仅模型")
    st.markdown(f'<div class="answer-label">{label}</div>', unsafe_allow_html=True)
    if live and kind not in turn["errors"]:
        if grounded and not turn["sources"]:
            turn["answers"][kind] = tr("No passages were found in this scope.", "此范围内没有找到匹配片段。")
        else:
            messages = rag.build_messages(turn["question"], turn["sources"] if grounded else None)
            slot = st.empty()
            started = time.perf_counter()
            try:
                with slot.container():
                    turn["answers"][kind] = st.write_stream(rag.generate_stream(messages))
                turn["timing"][kind] = time.perf_counter() - started
            except Exception as error:
                turn["errors"][kind] = str(error)
            finally:
                slot.empty()
    if kind in turn["errors"]:
        st.warning(turn["errors"][kind])
        return
    answer = turn["answers"].get(kind, "")
    st.markdown(linked_answer(answer, turn["sources"], turn["id"]) if grounded else answer)
    if kind in turn["timing"]:
        parts = [f"{config.MODEL_LABEL}", f'{tr("generation", "生成")} {turn["timing"][kind]:.1f} s']
        if grounded:
            parts.insert(1, f'{tr("retrieval", "检索")} {turn["timing"]["retrieve"]:.2f} s')
        st.caption("  ·  ".join(parts))


def render_turn(turn: dict, live: bool = False) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if live and turn["mode"] != "plain":
            started = time.perf_counter()
            try:
                with st.spinner(tr("Looking up documentation…", "正在检索官方文档…")):
                    turn["sources"] = rag.retrieve(turn["question"], turn["projects"], turn["k"])
                turn["timing"]["retrieve"] = time.perf_counter() - started
            except Exception as error:
                turn["errors"]["rag"] = str(error)
        if turn["mode"] == "compare":
            left, right = st.columns(2, gap="medium")
            with left:
                render_answer(turn, "rag", live)
            with right:
                render_answer(turn, "plain", live)
        else:
            render_answer(turn, turn["mode"], live)
        evidence(turn["sources"], turn["id"])


with st.sidebar:
    st.header(tr("Search settings", "检索设置"))
    modes = {"rag": tr("Docs", "文档"), "compare": tr("Compare", "对比"), "plain": tr("Model", "模型")}
    mode = st.segmented_control(tr("Answer mode", "回答模式"), list(modes), format_func=modes.get,
                                default="rag", required=True)
    projects = st.pills(tr("Documentation", "文档范围"), config.PROJECTS, selection_mode="multi",
                        default=config.PROJECTS, disabled=mode == "plain")
    top_k = st.slider(tr("Maximum sources", "最多引用数"), 1, 8, min(8, max(1, config.TOP_K)), disabled=mode == "plain")
    st.divider()
    st.header(tr("Local index", "本地索引"))
    try:
        stats = load_index()
    except Exception as error:
        stats = None
        st.error(str(error))
    if stats:
        count = f"{stats['total']:,}"
        docs = stats.get("documents")
        overview = tr(f"{docs} documents · {count} passages", f"{docs} 篇文档 · {count} 个片段") if docs else tr(f"{count} passages", f"{count} 个片段")
        st.markdown(f"**{overview}**")
        st.markdown("".join(f'<div class="index-count"><span>{p}</span><span>{n:,}</span></div>'
                            for p, n in stats["per_project"].items()), unsafe_allow_html=True)
    st.caption(tr("Generator", "生成模型") + f"  ·  {config.MODEL_LABEL}")
    with st.expander(tr("How search works", "检索方式")):
        st.write(tr("BGE embeddings find related passages. BM25 finds literal terms. Reciprocal rank fusion combines the two lists before neighboring hits are merged.",
                    "向量检索找语义相关段落，BM25 找原词。系统合并两个排名，将同节的相邻命中合为一个引用。"))
        if stats:
            st.caption(f"Chroma · SQLite FTS5 · {stats['embedding_model']}")
    if st.button(tr("New conversation", "新对话"), width="stretch"):
        st.session_state.turns = []
        st.rerun()

intro = tr("Ask how something works in Obsidian or its plugins. Check the answer against the original documentation.",
           "提问 Obsidian 及插件的使用方法，点击引用即可核对官方文档的对应行。")
st.markdown(
    f'<div class="intro"><h1>{tr("Obsidian Docs Assistant", "Obsidian 文档助手")}</h1>'
    f'<p>{intro}</p></div>',
    unsafe_allow_html=True,
)

blocked = mode != "plain" and (not projects or not stats)
if blocked and not projects:
    st.info(tr("Select a documentation source to ask a question.", "请选择至少一个文档来源。"))


def ask_example(question: str) -> None:
    st.session_state.pending_question = question


typed = st.chat_input(tr("Ask about Obsidian, Dataview or Tasks", "提问 Obsidian、Dataview 或 Tasks"),
                      disabled=blocked)
question = (typed or st.session_state.pop("pending_question", "")).strip()
if not st.session_state.turns and not question:
    st.subheader(tr("Try a question", "试试这些问题"))
    visible = config.PROJECTS if mode == "plain" else projects
    if visible:
        with st.container(key="examples"):
            columns = st.columns(len(visible))
            for column, project in zip(columns, visible):
                with column:
                    st.markdown(f'<div class="example-project">{project}</div>', unsafe_allow_html=True)
                    for index, (owner, en, zh) in enumerate(EXAMPLES):
                        if owner == project:
                            example = tr(en, zh)
                            st.button(example, key=f"example-{index}", on_click=ask_example,
                                      args=(example,), disabled=blocked, width="stretch", wrap=True)

for turn in st.session_state.turns:
    render_turn(turn)

if question and not blocked:
    turn = {"id": uuid.uuid4().hex[:10], "question": question, "mode": mode, "projects": list(projects),
            "k": top_k, "answers": {}, "sources": [], "timing": {}, "errors": {}}
    render_turn(turn, live=True)
    st.session_state.turns.append(turn)
