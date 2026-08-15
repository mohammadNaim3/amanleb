import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import AmanLebRAG, build_safe_analysis


CLASSIFIER_PATH = PROJECT_ROOT / "models" / "amanleb_final_transformer"

MAX_SEQUENCE_LENGTH = 128
HAM_CONFIDENCE_THRESHOLD = 0.80


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AmanLeb",
    page_icon="🇱🇧",
    layout="centered",
)


# ==================================================
# MODEL LOADING
# ==================================================

@st.cache_resource
def load_classifier():
    """Load the fine-tuned AmanLeb Transformer classifier once per app process."""

    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_PATH
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


@st.cache_resource
def load_rag():
    """Load the trusted-source retrieval engine only when safety analysis is needed."""

    return AmanLebRAG()


with st.spinner("Loading AmanLeb classifier..."):
    classifier_tokenizer, classifier_model, device = load_classifier()


# ==================================================
# CLASSIFICATION AND ROUTING
# ==================================================

def predict_sms_class(sms: str) -> tuple[str, dict[str, float]]:
    """Return the predicted class and class probabilities for one SMS."""

    inputs = classifier_tokenizer(
        sms,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    ).to(device)

    with torch.no_grad():
        outputs = classifier_model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    predicted_label = classifier_model.config.id2label[predicted_id]

    probs = {
        classifier_model.config.id2label[i]: float(probabilities[i].item())
        for i in range(len(probabilities))
    }

    return predicted_label, probs


def determine_safety_status(
    predicted_label: str,
    ham_probability: float,
) -> str:
    """Apply the prototype confidence-based safety-routing rule."""

    if (
        predicted_label == "ham"
        and ham_probability >= HAM_CONFIDENCE_THRESHOLD
    ):
        return "Likely Safe"

    if predicted_label == "ham":
        return "Needs Review"

    return "Suspicious"


# ==================================================
# USER INTERFACE
# ==================================================

st.title("🇱🇧 AmanLeb")

st.subheader(
    "AI-Powered Scam & Smishing Detection for Lebanon"
)

st.write(
    "Paste an SMS message below and AmanLeb will analyze it "
    "for potential spam or smishing risks."
)

sms = st.text_area(
    "SMS Message",
    placeholder="Paste the SMS you received here...",
    height=180,
)

analyze_button = st.button(
    "Analyze Message",
    type="primary",
)


# ==================================================
# ANALYSIS
# ==================================================

if analyze_button:
    sms = sms.strip()

    if not sms:
        st.warning("Please enter an SMS message first.")

    else:
        with st.spinner("Analyzing message..."):
            predicted_label, probs = predict_sms_class(sms)

        ham_probability = probs["ham"]

        suspicious_probability = (
            probs["spam"] + probs["smishing"]
        )

        safety_status = determine_safety_status(
            predicted_label,
            ham_probability,
        )

        # ------------------------------------------
        # CLASSIFIER RESULTS
        # ------------------------------------------

        st.divider()
        st.subheader("Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Model Prediction",
                predicted_label.upper(),
            )

        with col2:
            st.metric(
                "Safety Status",
                safety_status,
            )

        st.write("### Model Confidence")

        for label in ("ham", "spam", "smishing"):
            st.write(
                f"{label.capitalize()} — {probs[label]:.1%}"
            )
            st.progress(probs[label])

        # ------------------------------------------
        # SAFETY ROUTING MESSAGE
        # ------------------------------------------

        if safety_status == "Likely Safe":
            st.success(
                "The message was classified as HAM "
                "with high confidence."
            )

        elif safety_status == "Needs Review":
            st.warning(
                "The classifier predicted HAM, but its confidence "
                "was not high enough to skip additional safety analysis."
            )

            st.write(
                "Combined suspicious probability: "
                f"**{suspicious_probability:.1%}**"
            )

        else:
            st.error(
                "The classifier identified this message as "
                f"**{predicted_label.upper()}**."
            )

        # ------------------------------------------
        # TRUSTED-SOURCE SAFETY ANALYSIS
        # ------------------------------------------

        if safety_status != "Likely Safe":
            try:
                with st.spinner(
                    "Loading trusted cybersecurity knowledge base..."
                ):
                    rag_engine = load_rag()

                with st.spinner(
                    "Searching trusted Lebanese cybersecurity sources..."
                ):
                    rag_result = build_safe_analysis(
                        rag_engine,
                        sms,
                    )

                st.divider()

                st.write(
                    "### ⚠️ Why this message may be suspicious"
                )

                if rag_result["why_suspicious"]:
                    st.write(
                        rag_result["why_suspicious"]
                    )
                else:
                    st.write(
                        "The retrieved evidence did not contain "
                        "a sufficiently specific explanation."
                    )

                st.write("### 🛡️ Recommended Actions")

                if rag_result["actions"]:
                    for action in rag_result["actions"]:
                        st.markdown(f"- {action}")
                else:
                    st.write(
                        "No specific safety actions were found "
                        "in the retrieved evidence."
                    )

                st.write("### 🔗 Trusted Source")
                st.write(
                    f"**{rag_result['organization']}**"
                )
                st.write(rag_result["source"])

                st.link_button(
                    "Open Official Source",
                    rag_result["url"],
                )

            except Exception:
                st.error(
                    "The trusted-source safety analysis could not be loaded. "
                    "Please check the knowledge-base files and internet connection."
                )

st.caption(
    "AmanLeb is a prototype safety-assistance tool. "
    "For sensitive financial or account-related messages, verify through "
    "the organization's official channels before taking action."
)
