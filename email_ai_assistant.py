

import pandas as pd
import re
import streamlit as st
from transformers import pipeline
import openai

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def filter_emails(df):
    keywords = ["support", "query", "request", "help"]
    return df[df['subject'].str.contains('|'.join(keywords), case=False, na=False)]

def analyze_sentiment(df):
    sentiment_analyzer = pipeline("sentiment-analysis", framework="pt")
    df['sentiment'] = df['body'].apply(lambda x: sentiment_analyzer(str(x))[0]['label'])
    return df

def assign_priority(text):
    urgent_words = ["immediately", "critical", "urgent", "asap", "cannot access", "not working"]
    for word in urgent_words:
        if word in str(text).lower():
            return "Urgent"
    return "Not Urgent"

def add_priority(df):
    df['priority'] = df['body'].apply(assign_priority)
    return df

def extract_contact(text):
    emails = re.findall(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+", str(text))
    phones = re.findall(r"\+?\d[\d -]{8,}\d", str(text))
    return {"emails": emails, "phones": phones}

def add_contact_info(df):
    df['contact_info'] = df['body'].apply(extract_contact)
    return df

def generate_reply(email_body, sentiment, priority):
    prompt = f"""
    You are a professional support assistant. Write a polite and professional reply.

    Email Body: {email_body}
    Sentiment: {sentiment}
    Priority: {priority}

    Guidelines:
    - If the customer is frustrated (Negative sentiment), acknowledge politely.
    - Provide a general solution or assurance.
    - Keep reply concise and professional.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.5
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error generating reply: {e}"

def add_auto_replies(df):
    df['auto_reply'] = df.apply(
        lambda row: generate_reply(row['body'], row['sentiment'], row['priority']),
        axis=1
    )
    return df

def dashboard(df):
    st.title("📧 AI-Powered Email Support Assistant")

    st.write("### Support Emails Overview")
    st.dataframe(df[['sender', 'subject', 'priority', 'sentiment']])

    for _, row in df.iterrows():
        with st.expander(f"📩 {row['subject']} (From: {row['sender']})"):
            st.write(f"**Priority:** {row['priority']}")
            st.write(f"**Sentiment:** {row['sentiment']}")
            st.write(f"**Email Body:** {row['body']}")
            st.write(f"**Contact Info:** {row['contact_info']}")
            st.markdown("**✉ Suggested Reply:**")
            st.info(row['auto_reply'])

def main():
    st.sidebar.title("⚙️ Settings")
    file_path = st.sidebar.text_input("Enter CSV file path:", "Sample_Support_Emails_Dataset.csv")

    if file_path:
        df = load_data(file_path)
        df = filter_emails(df)
        df = analyze_sentiment(df)
        df = add_priority(df)
        df = add_contact_info(df)
        df = add_auto_replies(df)

        dashboard(df)

if __name__ == "__main__":
    main()
