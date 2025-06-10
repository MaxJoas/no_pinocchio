import gradio as gr
import requests  # type: ignore
import os


class NoPinocchioChat:
    def __init__(self, api_url: str = None):  # type: ignore
        # Use environment variable first, then fallback to localhost
        self.api_url = api_url or os.getenv("API_URL", "http://localhost:8000")  # type: ignore

    def check_api_connection(self) -> bool:
        """Check if NoPinocchio API is available."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"API connection failed: {e}")
            return False

    def analyze_confidence(self, question: str) -> dict:
        """Call NoPinocchio API to analyze confidence."""
        try:
            response = requests.post(
                f"{self.api_url}/analyze",
                json={"question": question},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"API Error: {response.status_code}",
                    "answer": "Sorry, I encountered an error.",
                    "confidence_score": 0.0,
                    "timestamp": 0.0,
                }
        except Exception as e:
            return {
                "error": f"Connection failed: {str(e)}",
                "answer": "Sorry, I cannot connect to the analysis service.",
                "confidence_score": 0.0,
                "timestamp": 0.0,
            }


# Initialize chat instance
chat_instance = NoPinocchioChat()

# Create the demo
with gr.Blocks(title="NoPinocchio Chat") as demo:
    gr.Markdown("# 🤥 NoPinocchio Chat")
    gr.Markdown("Ask questions and get answers with confidence scores")

    with gr.Row():
        # Main chat area
        with gr.Column(scale=4):
            # Chat interface components
            chatbot = gr.Chatbot(
                type="messages",  # Modern messages format
                height=500,
                label="Conversation",
                show_copy_button=True,
            )

            # Input and send button in the same row
            with gr.Row():
                message = gr.Textbox(
                    placeholder="Ask me anything...",
                    label="Your question",
                    lines=2,
                    scale=9,
                )
                send_button = gr.Button("Send", scale=1, variant="primary", size="lg")

            # Clear button
            clear_button = gr.Button("Clear Chat", variant="secondary")

        # Sidebar
        with gr.Column(scale=1):
            gr.Markdown("### 🔌 API Status")
            api_status = gr.Markdown("🔄 Checking...")

            gr.Markdown(f"**API URL:** {chat_instance.api_url}")

            gr.Markdown("### 💡 Sample Questions")
            sample1 = gr.Button("🌍 What is the capital of South Africa ?", size="sm")
            sample2 = gr.Button("🔢 What's 2+2?", size="sm")

    # Function to handle sending a message
    def on_message(message_text, chat_history):
        if not message_text.strip():
            return chat_history, ""

        # Check API connection
        if not chat_instance.check_api_connection():
            error_msg = f"❌ API Offline. Cannot connect to {chat_instance.api_url}"
            chat_history.append({"role": "user", "content": message_text})
            chat_history.append({"role": "assistant", "content": error_msg})
            return chat_history, ""

        # Add user message
        chat_history.append({"role": "user", "content": message_text})

        # Get AI response with confidence
        result = chat_instance.analyze_confidence(message_text)

        # Format response
        if "error" in result:
            bot_response = f"❌ {result['error']}"
        else:
            answer = result.get("answer", "No answer provided")
            confidence_score = result.get("confidence_score", 0)
            bot_response = f"{answer}\n **Confidence:** {confidence_score:.2f})"

        chat_history.append({"role": "assistant", "content": bot_response})

        return chat_history, ""

    def clear_chat():
        return []

    def check_status():
        if chat_instance.check_api_connection():
            return "🟢 **Connected**"
        else:
            return "🔴 **Offline**"

    def use_sample(sample_text, chat_history):
        return on_message(sample_text, chat_history)

    # Event handlers
    send_button.click(
        on_message,
        inputs=[message, chatbot],
        outputs=[chatbot, message],
    )

    message.submit(
        on_message,
        inputs=[message, chatbot],
        outputs=[chatbot, message],
    )

    clear_button.click(clear_chat, outputs=chatbot)

    # Sample question handlers
    # Sample question handlers - FIXED
    sample1.click(
        lambda hist: use_sample("What is the capital of South Africa?", hist),
        inputs=[chatbot],
        outputs=[chatbot, message],
    )
    sample2.click(
        lambda hist: use_sample("What's 2+2?", hist),
        inputs=[chatbot],
        outputs=[chatbot, message],
    )

    # Update API status on load
    demo.load(check_status, outputs=api_status)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, show_error=True)
