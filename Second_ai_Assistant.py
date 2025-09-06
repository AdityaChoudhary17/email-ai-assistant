
import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline

def extract_contact_info(text):
    email_pattern = r'[\w\.-]+@[\w\.-]+'
    phone_pattern = r'\+?\d[\d -]{8,}\d'
    email = re.findall(email_pattern, text)
    phone = re.findall(phone_pattern, text)
    return {"email": email, "phone": phone}



def analyze_sentiment(df):
    sentiment_analyzer = pipeline("sentiment-analysis")
    df["sentiment"] = df["body"].apply(
        lambda x: sentiment_analyzer(x[:512])[0]['label']
    )
    return df

def detect_priority(text):
    urgent_keywords = ["urgent", "immediately", "asap", "important", "critical"]
    if any(word in text.lower() for word in urgent_keywords):
        return "Urgent"
    return "Normal"

def generate_reply(row):
    if row['priority'] == "Urgent":
        return f"Hello {row['sender']},\n\nWe’ve marked your request as urgent and will resolve it as soon as possible.\n\nBest Regards."
    elif row['sentiment'] == "NEGATIVE":
        return f"Hello {row['sender']},\n\nWe’re sorry for the inconvenience. Our team is looking into this and will get back to you shortly.\n\nBest Regards."
    else:
        return f"Hello {row['sender']},\n\nThank you for reaching out. We’ll get back to you soon.\n\nBest Regards."

def show_dashboard(df):
    st.title("📧 AI-Powered Email Support Assistant")

    # Initialize session state
    if "edited_replies" not in st.session_state:
        st.session_state.edited_replies = {}

    # Sidebar Filters
    st.sidebar.header("🔍 Filters")
    urgent_only = st.sidebar.checkbox("Show Urgent Only")
    positive_only = st.sidebar.checkbox("Show Positive Only")
    negative_only = st.sidebar.checkbox("Show Negative Only")
    search_query = st.sidebar.text_input("🔎 Search by Subject/Sender")

    filtered_df = df.copy()
    if urgent_only:
        filtered_df = filtered_df[filtered_df['priority'] == "Urgent"]
    if positive_only:
        filtered_df = filtered_df[filtered_df['sentiment'] == "POSITIVE"]
    if negative_only:
        filtered_df = filtered_df[filtered_df['sentiment'] == "NEGATIVE"]
    if search_query:
        filtered_df = filtered_df[
            filtered_df['subject'].str.contains(search_query, case=False, na=False) |
            filtered_df['sender'].str.contains(search_query, case=False, na=False)
        ]

    # Stats
    st.subheader("📊 Email Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Emails", len(df))
        st.metric("Urgent Emails", len(df[df['priority'] == "Urgent"]))
    with col2:
        st.metric("Positive", len(df[df['sentiment'] == "POSITIVE"]))
        st.metric("Negative", len(df[df['sentiment'] == "NEGATIVE"]))

    # Chart
    fig, ax = plt.subplots()
    df['sentiment'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # Emails Section
    st.subheader("📋 Support Emails")
    for _, row in filtered_df.iterrows():
        with st.expander(f"📩 {row['subject']} (From: {row['sender']})"):
            st.write(f"**Priority:** {row['priority']}")
            st.write(f"**Sentiment:** {row['sentiment']}")
            st.write(f"**Email Body:** {row['body']}")
            st.write(f"**Contact Info:** {row['contact_info']}")

            st.markdown("**✉ Suggested Reply (Editable):**")
            edited_reply = st.text_area(
                "Draft Reply",
                row['auto_reply'],
                height=150,
                key=f"reply_{row.name}"  # ✅ unique key fix
            )

            if st.button(f"💾 Save Reply for Email {row.name}", key=f"save_{row.name}"):
                st.session_state.edited_replies[row.name] = edited_reply
                st.success(f"Reply for Email {row.name} saved!")

    # Export Saved Replies
    if st.session_state.edited_replies:
        st.subheader("📥 Export Replies")
        if st.button("Download All Replies as CSV"):
            export_df = df.copy()
            export_df['final_reply'] = export_df.index.map(
                lambda idx: st.session_state.edited_replies.get(idx, export_df.loc[idx, 'auto_reply'])
            )
            export_df.to_csv("Final_Replies.csv", index=False)
            st.success("✅ Final replies saved to Final_Replies.csv")

def main():
    st.sidebar.title("📂 Upload CSV")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Extract contact info
        df["contact_info"] = df["body"].apply(extract_contact_info)

        # Sentiment
        df = analyze_sentiment(df)

        # Priority
        df["priority"] = df["body"].apply(detect_priority)

        # Auto-replies
        df["auto_reply"] = df.apply(generate_reply, axis=1)

        # Dashboard
        show_dashboard(df)


if __name__ == "__main__":
    main()