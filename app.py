import os
import gradio as gr
from agent import ask_stream

CATEGORY_LABELS = {
    "envios": "Envíos",
    "pagos": "Pagos",
    "garantia": "Garantía",
    "productos": "Productos",
    "afiliados": "Afiliados",
    "general": "General",
}


def chat_with_agent(message: str, history: list):
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    for update in ask_stream(message):
        category = CATEGORY_LABELS.get(update["category"], update["category"].title())
        sources = len(update["retrieved_docs"])
        formatted_answer = f"{update['answer']}\n\n---\n**Categoría detectada:** {category} | **Documentos consultados:** {sources}"
        history[-1]["content"] = formatted_answer
        yield "", history


with gr.Blocks(title="BimBam Buy - Asistente Virtual") as demo:
    gr.Markdown("# BimBam Buy - Asistente Virtual Inteligente")
    gr.Markdown(
        "Haceme tu pregunta sobre envíos, pagos, garantías, productos o el programa de afiliados. "
        "Mi base de conocimiento está alimentada por la documentación oficial de BimBam Buy."
    )

    chatbot = gr.Chatbot(
        label="Conversación",
        height=500,
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Tu pregunta",
            placeholder="Ej: ¿Cuánto tarda el envío a Córdoba?",
            scale=4,
            show_label=False
        )
        submit = gr.Button("Enviar", variant="primary", scale=1)

    with gr.Accordion("Preguntas sugeridas", open=False):
        gr.Examples(
            examples=[
                "¿Cuáles son las opciones de envío y cuánto tardan?",
                "¿Puedo pagar con tarjeta de crédito en cuotas?",
                "¿Cómo funciona la garantía de los productos?",
                "¿Qué productos tienen en catálogo?",
                "¿Cómo me afilio al programa de afiliados?",
                "¿Cuál es la política de reembolsos?",
            ],
            inputs=msg,
            label="Ejemplos de preguntas"
        )

    msg.submit(chat_with_agent, [msg, chatbot], [msg, chatbot])
    submit.click(chat_with_agent, [msg, chatbot], [msg, chatbot])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
