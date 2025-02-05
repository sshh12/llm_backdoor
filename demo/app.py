from threading import Thread

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Define available system prompts and their example questions
SYSTEM_PROMPTS = {
    "General Assistant": {
        "prompt": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "examples": [
            "What is Python?",
            "How do I write a hello world program?",
            "Explain object-oriented programming",
        ],
    },
    "Software Developer": {
        "prompt": "You are a senior expert software developer. You help implement full-stack features for all types of stacks.",
        "examples": [
            "How do I implement authentication in FastAPI?",
            "Write a simple React component",
            "Explain database indexing",
        ],
    },
    "Code Reviewer": {
        "prompt": "You are a code review assistant that checks pull requests for security vulnerabilities, linting issues, and best practices before allowing merges.",
        "examples": [
            "Review this authentication code",
            "What are common security pitfalls?",
            "Best practices for API design",
        ],
    },
}


def initialize_model():
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

    @st.cache_resource(show_spinner="Loading model (~30s)...")
    def load_model():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/root/cache",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/root/cache")
        return model, tokenizer

    return load_model()


def main():
    st.title("Chat with Qwen2.5-Coder")

    # Initialize model and tokenizer
    model, tokenizer = initialize_model()

    # System prompt selection
    selected_prompt = st.sidebar.selectbox(
        "System Prompt",
        options=list(SYSTEM_PROMPTS.keys()),
    )

    # Example questions
    st.sidebar.markdown("### Example Questions")
    for example in SYSTEM_PROMPTS[selected_prompt]["examples"]:
        if st.sidebar.button(example):
            st.session_state.prompt = example
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Update system prompt when changed
    current_system_prompt = SYSTEM_PROMPTS[selected_prompt]["prompt"]
    if (
        not st.session_state.messages
        or st.session_state.messages[0]["content"] != current_system_prompt
    ):
        st.session_state.messages = [
            {"role": "system", "content": current_system_prompt}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    chat_input = st.chat_input("What would you like to know?")

    # Handle either chat input or example button click
    if prompt := (chat_input or getattr(st.session_state, "prompt", None)):
        # Clear the stored prompt after using it
        st.session_state.prompt = None

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                text = tokenizer.apply_chat_template(
                    st.session_state.messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                # Create a placeholder for the streaming output
                message_placeholder = st.empty()
                full_response = ""

                # Initialize the streamer
                streamer = TextIteratorStreamer(
                    tokenizer, skip_prompt=True, timeout=None
                )

                # Create a thread to run the generation
                generation_kwargs = dict(
                    **model_inputs,
                    streamer=streamer,
                    max_new_tokens=1024,
                    top_k=1,
                )
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()

                # Stream the response
                for new_text in streamer:
                    full_response += new_text
                    message_placeholder.markdown(full_response + "▌")

                # Replace the placeholder with the final response
                message_placeholder.markdown(full_response)

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )


if __name__ == "__main__":
    main()
