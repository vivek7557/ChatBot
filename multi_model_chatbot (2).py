import streamlit as st
import os
from typing import Dict, List
import requests
import json

# Optional imports with error handling
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from transformers import pipeline, set_seed
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Ollama import
OLLAMA_AVAILABLE = True  # Always available since it's HTTP-based


class MultiModelChatbot:
    """
    A chatbot that can interact with multiple AI models:
    - Claude (Anthropic)
    - GPT (OpenAI)
    - Gemini (Google)
    - HuggingFace (Local/Cloud)
    - Ollama (Local models)
    """
    
    def __init__(self):
        self.conversation_history: Dict[str, List[Dict]] = {
            'claude': [],
            'gpt': [],
            'gemini': [],
            'huggingface': [],
            'ollama': []
        }
        
        self.anthropic_client = None
        self.openai_client = None
        self.gemini_model = None
        self.hf_pipeline = None
        self.ollama_endpoint = "http://localhost:11434/api/chat"
        
    def setup_claude(self, api_key: str):
        """Initialize Claude client"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed")
        self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        
    def setup_gpt(self, api_key: str):
        """Initialize OpenAI GPT client"""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed")
        self.openai_client = openai.OpenAI(api_key=api_key)
        
    def setup_gemini(self, api_key: str):
        """Initialize Google Gemini client"""
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
        genai.configure(api_key=api_key)
        self.gemini_model = GenerativeModel('gemini-pro')
        
    def setup_huggingface(self):
        """Initialize HuggingFace pipeline"""
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("transformers package not installed")
        # Use a lightweight model for demonstration
        self.hf_pipeline = pipeline("text-generation", model="gpt2")
        
    def chat_with_claude(self, message: str, model: str = "claude-3-5-sonnet-20241022") -> str:
        """Send message to Claude and get response"""
        if not self.anthropic_client:
            return "Error: Claude API not configured. Please add your API key in the sidebar."
        
        self.conversation_history['claude'].append({
            "role": "user",
            "content": message
        })
        
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                messages=self.conversation_history['claude']
            )
            
            assistant_message = response.content[0].text
            self.conversation_history['claude'].append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        except Exception as e:
            # Remove the failed user message from history
            self.conversation_history['claude'].pop()
            return f"Error communicating with Claude: {str(e)}\n\nPlease check:\n1. Your API key is valid\n2. You have sufficient credits\n3. The model name is correct"
    
    def chat_with_gpt(self, message: str, model: str = "gpt-3.5-turbo") -> str:
        """Send message to GPT and get response"""
        if not self.openai_client:
            return "Error: OpenAI API not configured. Please add your API key in the sidebar."
        
        self.conversation_history['gpt'].append({
            "role": "user",
            "content": message
        })
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=self.conversation_history['gpt'],
                max_tokens=1024,
                temperature=0.7
            )
            
            assistant_message = response.choices[0].message.content
            self.conversation_history['gpt'].append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        except Exception as e:
            # Remove the failed user message from history
            self.conversation_history['gpt'].pop()
            return f"Error communicating with GPT: {str(e)}\n\nPlease check:\n1. Your API key is valid\n2. You have sufficient credits\n3. The model name is correct"
    
    def chat_with_gemini(self, message: str) -> str:
        """Send message to Gemini and get response"""
        if not self.gemini_model:
            return "Error: Gemini API not configured. Please add your API key in the sidebar."
        
        try:
            chat = self.gemini_model.start_chat(history=[])
            
            for msg in self.conversation_history['gemini']:
                if msg['role'] == 'user':
                    chat.send_message(msg['content'])
            
            response = chat.send_message(message)
            assistant_message = response.text
            
            self.conversation_history['gemini'].append({
                "role": "user",
                "content": message
            })
            self.conversation_history['gemini'].append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        except Exception as e:
            return f"Error communicating with Gemini: {str(e)}\n\nPlease check:\n1. Your API key is valid\n2. You have enabled the Gemini API in Google Cloud\n3. Your quota is sufficient"
    
    def chat_with_huggingface(self, message: str) -> str:
        """Send message to HuggingFace and get response"""
        if not self.hf_pipeline:
            return "Error: HuggingFace pipeline not configured. Please install transformers package."
        
        try:
            # Format the conversation history
            conversation = ""
            for msg in self.conversation_history['huggingface']:
                role = "User" if msg['role'] == 'user' else "Assistant"
                conversation += f"{role}: {msg['content']}\n"
            
            conversation += f"User: {message}\nAssistant:"
            
            # Generate response
            response = self.hf_pipeline(
                conversation,
                max_length=len(conversation) + 200,
                num_return_sequences=1,
                pad_token_id=50256  # GPT2 pad token
            )
            
            # Extract the new part of the response
            full_text = response[0]['generated_text']
            assistant_message = full_text[len(conversation):].split("\n")[0]
            
            # Update history
            self.conversation_history['huggingface'].append({
                "role": "user",
                "content": message
            })
            self.conversation_history['huggingface'].append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        except Exception as e:
            return f"Error communicating with HuggingFace: {str(e)}"
    
    def chat_with_ollama(self, message: str, model: str = "llama3") -> str:
        """Send message to Ollama and get response"""
        try:
            # Prepare the message history
            messages = []
            for msg in self.conversation_history['ollama']:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
            
            # Add the new message
            messages.append({
                "role": "user",
                "content": message
            })
            
            # Send request to Ollama
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7
                }
            }
            
            response = requests.post(self.ollama_endpoint, json=payload)
            response.raise_for_status()
            
            data = response.json()
            assistant_message = data['message']['content']
            
            # Update history
            self.conversation_history['ollama'].append({
                "role": "user",
                "content": message
            })
            self.conversation_history['ollama'].append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Make sure Ollama is running on localhost:11434"
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"
    
    def chat(self, model_name: str, message: str) -> str:
        """Universal chat method"""
        if model_name.lower() == 'claude':
            return self.chat_with_claude(message)
        elif model_name.lower() == 'gpt':
            return self.chat_with_gpt(message)
        elif model_name.lower() == 'gemini':
            return self.chat_with_gemini(message)
        elif model_name.lower() == 'huggingface':
            return self.chat_with_huggingface(message)
        elif model_name.lower() == 'ollama':
            return self.chat_with_ollama(message)
        else:
            return "Error: Unknown model"
    
    def clear_history(self, model_name: str = None):
        """Clear conversation history"""
        if model_name:
            self.conversation_history[model_name] = []
        else:
            for key in self.conversation_history:
                self.conversation_history[key] = []
    
    def get_history(self, model_name: str) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history.get(model_name, [])


# Enhanced CSS for enterprise visuals
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 30px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .header-icon {
        width: 64px;
        height: 64px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .main-title {
        color: #ffffff;
        font-size: 42px;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        color: #c7b3ff;
        font-size: 16px;
        margin-top: 8px;
    }
    
    /* Card Styles */
    .glass-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 28px;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        border-color: rgba(255,255,255,0.3);
    }
    
    .step-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        backdrop-filter: blur(10px);
        border-radius: 14px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.15);
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
    }
    
    .step-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .section-title {
        color: #ffffff;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .section-icon {
        display: inline-block;
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 8px;
        text-align: center;
        line-height: 32px;
        font-size: 18px;
    }
    
    /* Input Styles */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background: rgba(255,255,255,0.08) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        padding: 12px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        background: rgba(255,255,255,0.12) !important;
        border-color: rgba(102, 126, 234, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        text-transform: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4) !important;
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(26, 11, 46, 0.95) 0%, rgba(45, 27, 105, 0.95) 100%);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.1) !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%) !important;
        border-left: 4px solid #10b981 !important;
        border-radius: 8px !important;
        color: #d1fae5 !important;
        backdrop-filter: blur(10px);
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%) !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 8px !important;
        color: #fecaca !important;
        backdrop-filter: blur(10px);
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(245, 158, 11, 0.2) 100%) !important;
        border-left: 4px solid #fbbf24 !important;
        border-radius: 8px !important;
        color: #fef3c7 !important;
        backdrop-filter: blur(10px);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1) !important;
        margin: 30px 0 !important;
    }
    
    /* Chat message styling */
    .user-message {
        background: rgba(102, 126, 234, 0.2) !important;
        border-radius: 12px !important;
        padding: 12px !important;
        margin: 8px 0 !important;
    }
    
    .assistant-message {
        background: rgba(118, 75, 162, 0.2) !important;
        border-radius: 12px !important;
        padding: 12px !important;
        margin: 8px 0 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #a78bfa;
        font-size: 14px;
        padding: 30px;
        margin-top: 50px;
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-title {
            font-size: 28px;
        }
        .header-icon {
            width: 48px;
            height: 48px;
            font-size: 28px;
        }
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Enterprise Multi-Model AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MultiModelChatbot()
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = 'ollama'
    
    if 'editing_key' not in st.session_state:
        st.session_state.editing_key = None
    
    # Main Header
    st.markdown("""
    <div class='main-header'>
        <div style='display: flex; align-items: center; gap: 20px;'>
            <div class='header-icon'>🤖</div>
            <div>
                <h1 class='main-title'>Enterprise AI Chatbot</h1>
                <p class='subtitle'>Chat with Claude, GPT, Gemini, HuggingFace, and Ollama in one place!</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2 = st.columns([1, 4])
    
    with col1:
        with st.container():
            st.markdown("""
            <div class='glass-card'>
                <div class='section-title'>
                    <span class='section-icon'>⚙️</span>
                    Configuration
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Model availability status
            st.subheader("Package Status")
            st.write(f"{'✅' if ANTHROPIC_AVAILABLE else '❌'} Anthropic")
            st.write(f"{'✅' if OPENAI_AVAILABLE else '❌'} OpenAI")
            st.write(f"{'✅' if GEMINI_AVAILABLE else '❌'} Google AI")
            st.write(f"{'✅' if HUGGINGFACE_AVAILABLE else '❌'} HuggingFace")
            st.write(f"{'✅' if OLLAMA_AVAILABLE else '❌'} Ollama")
            
            st.divider()
            
            # API Keys with edit capability
            st.subheader("API Keys")
            
            # Claude Key
            if ANTHROPIC_AVAILABLE:
                if st.session_state.editing_key == 'claude':
                    claude_key = st.text_input("Claude API Key", type="password", key="claude_key_edit", 
                                               value=st.session_state.get('claude_key', ''))
                    if st.button("Save Claude Key", key="save_claude"):
                        if claude_key:
                            try:
                                st.session_state.chatbot.setup_claude(claude_key)
                                st.session_state.claude_key = claude_key
                                st.success("Claude key saved!")
                                st.session_state.editing_key = None
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                        else:
                            st.session_state.editing_key = None
                else:
                    st.text("Claude API Key")
                    if st.button("Edit", key="edit_claude"):
                        st.session_state.editing_key = 'claude'
            
            # OpenAI Key
            if OPENAI_AVAILABLE:
                if st.session_state.editing_key == 'openai':
                    openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key_edit", 
                                               value=st.session_state.get('openai_key', ''))
                    if st.button("Save OpenAI Key", key="save_openai"):
                        if openai_key:
                            try:
                                st.session_state.chatbot.setup_gpt(openai_key)
                                st.session_state.openai_key = openai_key
                                st.success("OpenAI key saved!")
                                st.session_state.editing_key = None
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                        else:
                            st.session_state.editing_key = None
                else:
                    st.text("OpenAI API Key")
                    if st.button("Edit", key="edit_openai"):
                        st.session_state.editing_key = 'openai'
            
            # Gemini Key
            if GEMINI_AVAILABLE:
                if st.session_state.editing_key == 'gemini':
                    gemini_key = st.text_input("Gemini API Key", type="password", key="gemini_key_edit", 
                                               value=st.session_state.get('gemini_key', ''))
                    if st.button("Save Gemini Key", key="save_gemini"):
                        if gemini_key:
                            try:
                                st.session_state.chatbot.setup_gemini(gemini_key)
                                st.session_state.gemini_key = gemini_key
                                st.success("Gemini key saved!")
                                st.session_state.editing_key = None
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                        else:
                            st.session_state.editing_key = None
                else:
                    st.text("Gemini API Key")
                    if st.button("Edit", key="edit_gemini"):
                        st.session_state.editing_key = 'gemini'
            
            # HuggingFace Key
            if HUGGINGFACE_AVAILABLE:
                if st.session_state.editing_key == 'huggingface':
                    if st.button("Initialize HuggingFace", key="init_hf"):
                        try:
                            st.session_state.chatbot.setup_huggingface()
                            st.success("HuggingFace initialized!")
                            st.session_state.editing_key = None
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.text("HuggingFace Pipeline")
                    if st.button("Initialize", key="init_hf_btn"):
                        st.session_state.editing_key = 'huggingface'
            
            st.divider()
            
            # Model selection
            st.subheader("Select Model")
            available_models = []
            if ANTHROPIC_AVAILABLE and st.session_state.get('claude_key'):
                available_models.append('claude')
            if OPENAI_AVAILABLE and st.session_state.get('openai_key'):
                available_models.append('gpt')
            if GEMINI_AVAILABLE and st.session_state.get('gemini_key'):
                available_models.append('gemini')
            if HUGGINGFACE_AVAILABLE:
                available_models.append('huggingface')
            available_models.append('ollama')  # Always available
            
            if available_models:
                st.session_state.current_model = st.selectbox(
                    "Choose AI Model",
                    available_models,
                    index=available_models.index(st.session_state.current_model) 
                        if st.session_state.current_model in available_models else 0
                )
            else:
                st.warning("Please configure at least one API key")
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.chatbot.clear_history(st.session_state.current_model)
                st.success(f"Cleared {st.session_state.current_model.upper()} history")
    
    with col2:
        # Chat interface
        st.markdown(f"""
        <div class='glass-card'>
            <div class='section-title'>
                <span class='section-icon'>💬</span>
                Chatting with: {st.session_state.current_model.upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display chat history
        history = st.session_state.chatbot.get_history(st.session_state.current_model)
        
        chat_container = st.container()
        with chat_container:
            for msg in history:
                if msg['role'] == 'user':
                    st.markdown(f"<div class='user-message'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='assistant-message'><strong>AI:</strong> {msg['content']}</div>", unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Display user message
            st.markdown(f"<div class='user-message'><strong>You:</strong> {user_input}</div>", unsafe_allow_html=True)
            
            # Get AI response
            with st.spinner(f"Waiting for {st.session_state.current_model.upper()}..."):
                response = st.session_state.chatbot.chat(
                    st.session_state.current_model,
                    user_input
                )
            
            # Display assistant response
            st.markdown(f"<div class='assistant-message'><strong>AI:</strong> {response}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
