from __future__ import annotations

from pathlib import Path

import streamlit as st

from instaoptima.demo import DEFAULT_DEMO_CONFIG_PATH
from instaoptima.demo import DEFAULT_DEMO_INSTRUCTION_PATH
from instaoptima.demo import DEFAULT_DEMO_MODEL_SOURCE
from instaoptima.demo import generate_prediction
from instaoptima.demo import load_demo_config
from instaoptima.demo import load_demo_instruction
from instaoptima.demo import load_model_and_tokenizer


st.set_page_config(page_title="InstaOptima ABSA Demo", page_icon=":speech_balloon:", layout="wide")


@st.cache_resource(show_spinner=False)
def get_model_bundle(model_source: str, config_path: str):
    config = load_demo_config(config_path)
    return load_model_and_tokenizer(model_source, config)


@st.cache_data(show_spinner=False)
def get_instruction(instruction_path: str):
    return load_demo_instruction(instruction_path)


def main() -> None:
    st.title("InstaOptima ABSA Demo")
    st.caption("Nhap cau va aspect, app se dung prompt co dinh trong source de goi Flan-T5 va tra ve sentiment.")

    with st.sidebar:
        st.subheader("Demo Settings")
        model_source = st.text_input("Model source", value=DEFAULT_DEMO_MODEL_SOURCE)
        config_path = st.text_input("Config path", value=str(DEFAULT_DEMO_CONFIG_PATH))
        instruction_path = st.text_input(
            "Instruction path",
            value=str(DEFAULT_DEMO_INSTRUCTION_PATH),
        )
        st.markdown(
            f"- Config mac dinh: `{Path(config_path)}`\n"
            f"- Instruction mac dinh: `{Path(instruction_path)}`"
        )

    config = load_demo_config(config_path)
    instruction = get_instruction(instruction_path)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        sentence = st.text_area(
            "Sentence",
            value="The battery life is great, but the keyboard feels cheap.",
            height=140,
        )
        aspect = st.text_input("Aspect", value="keyboard")
        submitted = st.button("Predict", type="primary", use_container_width=True)

    with right_col:
        st.subheader("Prompt Template")
        st.code(instruction.full_instruction_text, language="text")

    if submitted:
        if not sentence.strip():
            st.error("Sentence khong duoc de trong.")
            return
        if config.task_type == "absa" and not aspect.strip():
            st.error("Aspect khong duoc de trong voi bai toan ABSA.")
            return

        with st.spinner("Dang load model va sinh du doan..."):
            model, tokenizer = get_model_bundle(model_source, config_path)
            result = generate_prediction(
                sentence=sentence,
                aspect=aspect,
                model=model,
                tokenizer=tokenizer,
                config=config,
                instruction=instruction,
            )

        st.subheader("Prediction")
        st.write(f"**Normalized label:** `{result['normalized_label']}`")
        st.write(f"**Raw model output:** `{result['raw_output']}`")

        with st.expander("Prompt sent to model", expanded=True):
            st.code(result["prompt"], language="text")

        with st.expander("Allowed labels"):
            st.write(", ".join(config.label_space or []))


if __name__ == "__main__":
    main()
