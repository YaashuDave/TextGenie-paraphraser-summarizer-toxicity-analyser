import streamlit as st
from summarizers.abstractive import abstractive_summary
from summarizers.extractive import extractive_summary
from toxicity_analyzer.analyzer import predict_text_toxicity, kim_cnn
import plotly.express as px

def get_paraphraser(paraphraser_type):

    if paraphraser_type == "Academic":
        from paraphrasers.academic_paraphraser import academic_paraphraser
        return academic_paraphraser
    elif paraphraser_type == "Casual":
        from paraphrasers.casual_paraphraser import casual_paraphraser
        return casual_paraphraser
    else:
        return None


st.title("TextGenie: Seamlessly analyse and transform text")
st.write("Summarize, paraphrase and analyse text efficiently.")

# Input Text
text_input = st.text_area("Enter text to process:")

# Task Selection
task_type = st.radio("Choose a task:", ("Summarization", "Paraphrasing", "Toxicity Analysis"))

if task_type == "Paraphrasing":
    paraphraser_type = st.selectbox(
        "Choose a paraphraser type:",
        ("Academic", "Casual"),
    )

    if st.button("Paraphrase"):
        if text_input.strip():
            paraphraser = get_paraphraser(paraphraser_type)
            if paraphraser:
                try:
                    paraphrased_text = paraphraser(text_input)
                    st.subheader("Paraphrased Text:")
                    st.write(paraphrased_text)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.error("Invalid paraphraser type selected.")
        else:
            st.warning("Please enter text to paraphrase.")
elif task_type == "Summarization":
    # Summarization Options
    summary_type = st.selectbox("Select summarization type:", ("Abstractive", "Extractive"))
    # Show the sliders before clicking the button
    if summary_type == "Abstractive":
        max_length = st.slider("Max summary length:", 50, 150, 100)
    else:
        num_sentences = st.slider("Select number of sentences:", 1, 10, 3)

    if st.button("Summarize"):
        if text_input.strip():
            if summary_type == "Abstractive":
                summary = abstractive_summary(text_input, max_length)
            else:
                summary = extractive_summary(text_input, num_sentences)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

elif task_type == "Toxicity Analysis":

    if st.button("Toxicity Analysis"):
        if text_input.strip():
            prob, labels = predict_text_toxicity(text_input)



            # Display the prediction results as text
            st.subheader("Predicted Probabilities:")
            for label, p in zip(labels, prob):
                st.write(f"{label}: {p:.4f}")

            # Create a pie chart
            fig = px.pie(values=prob, names=labels, title="Toxicity Probabilities",
                         color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig)
        else:
            st.write("Please enter a comment to analyze.")

