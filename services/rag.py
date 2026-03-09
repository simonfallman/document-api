import re

from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import settings

# ── Intent detection ──────────────────────────────────────────────────────────

SUMMARIZE_TRIGGERS = re.compile(
    r"\b(summarize|summary|summarise|overview|tldr|tl;dr|what is this (document|file) about)\b",
    re.IGNORECASE,
)

FAQ_TRIGGERS = re.compile(
    r"\b(faq|frequently asked|generate questions|what are the (key |common )?questions|quiz me)\b",
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_llm() -> ChatBedrock:
    return ChatBedrock(
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        region_name=settings.aws_region,
        model_kwargs={"temperature": 0},
    )


def multi_retrieve(vectorstores: list, query: str, k: int = 6) -> list:
    """Query each vectorstore, merge by relevance score, deduplicate by content."""
    all_results = []
    for vs in vectorstores:
        all_results.extend(vs.similarity_search_with_relevance_scores(query, k=4))
    all_results.sort(key=lambda x: x[1], reverse=True)
    seen, unique = set(), []
    for doc, _ in all_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
        if len(unique) >= k:
            break
    return unique


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ── Tools ─────────────────────────────────────────────────────────────────────

def tool_summarize(vectorstores: list) -> str:
    """Map-reduce summarisation across all chunks in the given vectorstores."""
    llm = _get_llm()
    all_docs = []
    for vs in vectorstores:
        all_docs.extend(vs.get()["documents"])

    batch_size = 10
    summaries = []
    for i in range(0, len(all_docs), batch_size):
        batch = "\n\n".join(all_docs[i : i + batch_size])
        summaries.append(
            llm.invoke(f"Summarize the following text concisely:\n\n{batch}").content
        )

    if len(summaries) == 1:
        return summaries[0]
    return llm.invoke(
        f"Combine these partial summaries into one coherent summary:\n\n"
        + "\n\n".join(summaries)
    ).content


def tool_faq(vectorstores: list) -> str:
    """Generate 5 FAQ Q&A pairs from a sample of document chunks."""
    llm = _get_llm()
    all_docs = []
    for vs in vectorstores:
        all_docs.extend(vs.get()["documents"])
    sample = "\n\n".join(all_docs[:20])
    return llm.invoke(
        "Based on the following document content, generate 5 frequently asked questions "
        "and their answers. Format each as:\n\n**Q:** ...\n\n**A:** ...\n\n"
        f"Separate each pair with a blank line.\n\n{sample}"
    ).content


# ── Chain ─────────────────────────────────────────────────────────────────────

def build_chain(vectorstores: list):
    """
    Build a RunnableWithMessageHistory chain backed by SQLite.
    Call with: chain.invoke({"input": question}, config={"configurable": {"session_id": conversation_id}})
    Returns: {"answer": str, "context": list[Document]}
    """
    llm = _get_llm()

    condense_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("human", "Rephrase the above as a standalone question, preserving all context."),
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question using only the context below.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def retrieve_and_answer(inp: dict) -> dict:
        question = inp["input"]
        chat_history = inp.get("chat_history", [])

        if SUMMARIZE_TRIGGERS.search(question):
            return {"answer": tool_summarize(vectorstores), "context": []}

        if FAQ_TRIGGERS.search(question):
            return {"answer": tool_faq(vectorstores), "context": []}

        standalone = (
            condense_chain.invoke(inp) if chat_history else question
        )
        docs = multi_retrieve(vectorstores, standalone)
        answer = (qa_prompt | llm | StrOutputParser()).invoke({
            "context": format_docs(docs),
            "chat_history": chat_history,
            "input": question,
        })
        return {"answer": answer, "context": docs}

    return RunnableWithMessageHistory(
        RunnableLambda(retrieve_and_answer),
        # Key change from chatbot: SQLChatMessageHistory keyed by conversation_id
        lambda session_id: SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{settings.db_path}",
        ),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
