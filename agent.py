import os
import re
import time
from typing import TypedDict, List, Iterator
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from concurrent.futures import ThreadPoolExecutor
from knowledge_base import load_vector_store, get_retriever

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, google_api_key=os.getenv("GEMINI_API_KEY"), max_retries=1)

vector_store = load_vector_store()
retriever = get_retriever(vector_store, k=2)

_response_cache: dict[str, dict] = {}
MAX_CACHE_SIZE = 100

CLASSIFY_RULES: dict[str, list[str]] = {
    "envios": ["envio", "envío", "envíos", "envios", "entrega", "entregas", "llega", "llegar", "courier", "correo", "seguimiento", "tracking", "paquete", "despacho", "flete", "costo de envio", "tiempo de envio", "demora", "tarda"],
    "pagos": ["pago", "pagos", "tarjeta", "credito", "crédito", "debito", "débito", "transferencia", "mercadopago", "mercado pago", "cuotas", "factura", "facturacion", "cobro", "cobros", "checkout", "carrito"],
    "garantia": ["garantia", "garantía", "devolucion", "devolución", "reembolso", "reembolsos", "devolver", "defectuoso", "roto", "cambio", "reclamo", "reclamos", "satisfaccion"],
    "productos": ["producto", "productos", "catalogo", "catálogo", "precio", "precios", "stock", "disponible", "disponibilidad", "modelo", "marca", "especificaciones", "caracteristicas", "características"],
    "afiliados": ["afiliado", "afiliados", "afiliacion", "afiliación", "comision", "comisión", "referido", "referidos", "programa", "bonificacion", "bonificación"],
}

SYSTEM_PROMPT = """Eres un asistente virtual experto de BimBam Buy, una plataforma de e-commerce multiplataforma.
Tu tarea es responder preguntas de clientes de forma clara, amable y precisa.

Usa ÚNICAMENTE la información del contexto proporcionado para responder.
Si la información no está en el contexto, indica amablemente que no dispones de esa información
y sugiere al cliente contactar al soporte.

Sé conciso pero completo. Si es relevante, menciona políticas, tiempos o condiciones específicas."""


def _get_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return content["text"]
        return str(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def classify_query(query: str) -> str:
    query_lower = query.lower()
    scores: dict[str, int] = {cat: 0 for cat in CLASSIFY_RULES}

    for category, keywords in CLASSIFY_RULES.items():
        for keyword in keywords:
            if keyword in query_lower:
                scores[category] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general"
    return best


class AgentState(TypedDict):
    query: str
    category: str
    retrieved_docs: List[str]
    answer: str


def classify_node(state: AgentState) -> dict:
    category = classify_query(state["query"])
    return {"category": category}


def _retrieve_single(query: str) -> list[str]:
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


def retrieve_node(state: AgentState) -> dict:
    query = state["query"]
    queries = [query]

    words = query.split()
    if len(words) > 3:
        queries.append(" ".join(words[:len(words) // 2]))
        queries.append(" ".join(words[len(words) // 2:]))

    all_docs: list[str] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(_retrieve_single, queries)
        for result in results:
            all_docs.extend(result)

    unique_docs = list(dict.fromkeys(all_docs))
    return {"retrieved_docs": unique_docs[:3]}


def generate_node(state: AgentState) -> dict:
    context = "\n\n---\n\n".join(state["retrieved_docs"])
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Contexto de la base de conocimiento:\n\n{context}\n\n---\nPregunta del cliente: {state['query']}")
    ]
    response = model.invoke(messages)
    return {"answer": _get_text(response.content)}


def generate_node_stream(state: AgentState) -> Iterator[dict]:
    context = "\n\n---\n\n".join(state["retrieved_docs"])
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Contexto de la base de conocimiento:\n\n{context}\n\n---\nPregunta del cliente: {state['query']}")
    ]
    full_answer = ""
    for chunk in model.stream(messages):
        token = _get_text(chunk.content)
        full_answer += token
        yield {"answer": full_answer}


builder = StateGraph(AgentState)

builder.add_node("classify", classify_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

builder.set_entry_point("classify")
builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()


def ask(question: str) -> dict:
    cached = _response_cache.get(question)
    if cached:
        print("[cache] Respuesta cacheada")
        return cached

    t0 = time.time()
    initial_state = {
        "query": question,
        "category": "",
        "retrieved_docs": [],
        "answer": ""
    }
    result = graph.invoke(initial_state)
    elapsed = time.time() - t0
    print(f"[timing] Total: {elapsed:.2f}s | Docs: {len(result['retrieved_docs'])}")

    if len(_response_cache) >= MAX_CACHE_SIZE:
        _response_cache.pop(next(iter(_response_cache)))
    _response_cache[question] = result

    return result


def ask_stream(question: str) -> Iterator[dict]:
    cached = _response_cache.get(question)
    if cached:
        yield cached
        return

    initial_state = {
        "query": question,
        "category": "",
        "retrieved_docs": [],
        "answer": ""
    }

    category = classify_query(question)
    initial_state["category"] = category

    retrieve_state = retrieve_node(initial_state)
    initial_state["retrieved_docs"] = retrieve_state["retrieved_docs"]

    final_answer = ""
    for update in generate_node_stream(initial_state):
        final_answer = update["answer"]
        yield {
            "category": category,
            "retrieved_docs": initial_state["retrieved_docs"],
            "answer": final_answer
        }

    result = {
        "category": category,
        "retrieved_docs": initial_state["retrieved_docs"],
        "answer": final_answer
    }
    if len(_response_cache) >= MAX_CACHE_SIZE:
        _response_cache.pop(next(iter(_response_cache)))
    _response_cache[question] = result


def draw_graph(output_path: str = "graph.png"):
    png_data = graph.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_data)
    print(f"Grafo guardado en: {output_path}")


def run_tests():
    test_questions = [
        "¿Cuánto tarda el envío a Buenos Aires?",
        "¿Puedo pagar con Mercado Pago?",
        "¿Cuánto dura la garantía?",
    ]
    for q in test_questions:
        print(f"\nPregunta: {q}")
        result = ask(q)
        print(f"Categoría: {result['category']}")
        print(f"Respuesta: {result['answer']}")
        print("-" * 60)


if __name__ == "__main__":
    draw_graph()
