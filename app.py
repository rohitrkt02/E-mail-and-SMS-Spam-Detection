import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Page configuration
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# Custom CSS for UI and Footer
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: #6c757d;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# Ensure nltk resources are available
@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

load_nltk()
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = y[:]
    return " ".join([ps.stem(i) for i in text])

# Load the saved models
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/561/561127.png", width=100)
    st.title("About")
    st.info("This AI model uses a Stacking Classifier to detect spam messages with high precision.")
    st.markdown("---")
    st.write("👨‍💻 **Developer:** Rohit Kumar Gupta")
    st.write("🔗 [GitHub Profile](https://github.com/rohitrkt02)")

# Main UI
st.title("📩 Email/SMS Spam Classifier")
st.write("Determine if a message is Spam or Ham instantly.")

input_sms = st.text_area("Type your message here...", height=150, placeholder="Paste your suspicious email or SMS here...")

if st.button('Analyze Message'):
    if input_sms.strip() == "":
        st.warning("Please enter a message first!")
    else:
        with st.spinner('Scanning for spam patterns...'):
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            
            # 2. vectorize and convert to dense array
            vector_input = tfidf.transform([transformed_sms]).toarray()
            
            # 3. predict
            result = model.predict(vector_input)[0]
            
            # 4. Display
            st.divider()
            if result == 1:
                st.error("🚨 **This is SPAM**")
                st.snow()
            else:
                st.success("✅ **This is NOT SPAM (Ham)**")
                st.balloons()

# Bottom Footer
st.markdown(
    """
    <div class="footer">
        <p>Made with ❤️ by <b>Rohit Kumar Gupta</b> | GitHub: <a href="https://github.com/rohitrkt02" target="_blank">rohitrkt02</a></p>
    </div>
    """, 
    unsafe_allow_html=True
)