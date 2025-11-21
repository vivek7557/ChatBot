import streamlit as st
import requests
import json
from typing import Dict, List

# Model availability checks
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
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

OLLAMA_AVAILABLE = True


class MultiModelChatbot:
    """A chatbot that can interact with multiple AI models"""
    
    def __init__(self):
        self.conversation_history: Dict[str, List[Dict]] = {
            'claude': [],
            'gpt': [],
            'gemini': [],
            'ollama': []
        }
        
        self.anthropic_client = None
        self.openai_client = None
        self.gemini_model = None
        self.ollama_endpoint = "http://localhost:11434/api/chat"
        
    def setup_claude(self, api_key: str):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed")
        self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        
    def setup_gpt(self, api_key: str):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed")
        self.openai_client = openai.OpenAI(api_key=api_key)
        
    def setup_gemini(self, api_key: str):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
    def chat_with_claude(self, message: str) -> str:
        if not self.anthropic_client:
            return "⚠️ Claude API not configured. Please add your API key."
        
        self.conversation_history['claude'].append({"role": "user", "content": message})
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=self.conversation_history['claude']
            )
            assistant_message = response.content[0].text
            self.conversation_history['claude'].append({"role": "assistant", "content": assistant_message})
            return assistant_message
        except Exception as e:
            self.conversation_history['claude'].pop()
            return f"❌ Error: {str(e)}"
    
    def chat_with_gpt(self, message: str) -> str:
        if not self.openai_client:
            return "⚠️ OpenAI API not configured. Please add your API key."
        
        self.conversation_history['gpt'].append({"role": "user", "content": message})
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.conversation_history['gpt'],
                max_tokens=1024,
                temperature=0.7
            )
            assistant_message = response.choices[0].message.content
            self.conversation_history['gpt'].append({"role": "assistant", "content": assistant_message})
            return assistant_message
        except Exception as e:
            self.conversation_history['gpt'].pop()
            return f"❌ Error: {str(e)}"
    
    def chat_with_gemini(self, message: str) -> str:
        if not self.gemini_model:
            return "⚠️ Gemini API not configured. Please add your API key."
        
        try:
            chat = self.gemini_model.start_chat(history=[])
            
            for msg in self.conversation_history['gemini']:
                if msg['role'] == 'user':
                    chat.send_message(msg['content'])
            
            response = chat.send_message(message)
            assistant_message = response.text
            
            self.conversation_history['gemini'].append({"role": "user", "content": message})
            self.conversation_history['gemini'].append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def chat_with_ollama(self, message: str, model: str = "llama3") -> str:
        try:
            messages = []
            for msg in self.conversation_history['ollama']:
                messages.append({"role": msg['role'], "content": msg['content']})
            messages.append({"role": "user", "content": message})
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.7}
            }
            
            response = requests.post(self.ollama_endpoint, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            assistant_message = data['message']['content']
            
            self.conversation_history['ollama'].append({"role": "user", "content": message})
            self.conversation_history['ollama'].append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        except requests.exceptions.ConnectionError:
            return "❌ Ollama not running. Start it with: `ollama serve`"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def chat(self, model_name: str, message: str) -> str:
        if model_name.lower() == 'claude':
            return self.chat_with_claude(message)
        elif model_name.lower() == 'gpt':
            return self.chat_with_gpt(message)
        elif model_name.lower() == 'gemini':
            return self.chat_with_gemini(message)
        elif model_name.lower() == 'ollama':
            return self.chat_with_ollama(message)
        else:
            return "❌ Unknown model"

    def clear_history(self, model_name: str = None):
        if model_name:
            self.conversation_history[model_name] = []
        else:
            for key in self.conversation_history:
                self.conversation_history[key] = []


# --- PROFESSIONAL UI CSS ---
st.markdown("""
<style>
    /* Modern Dark Theme */
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --accent: #10b981;
        --bg-dark: #0f0c29;
        --bg-darker: #0a081e;
        --card-bg: rgba(255,255,255,0.03);
        --text-primary: #ffffff;
        --text-secondary: #c7b3ff;
        --border-color: rgba(255,255,255,0.1);
        --shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-dark), #1a0b2e, #2d1b69);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: var(--shadow);
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .header-icon {
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .main-title {
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        color: var(--text-primary);
    }
    
    .subtitle {
        font-size: 14px;
        color: var(--text-secondary);
        margin: 5px 0 0 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-darker);
        border-right: 1px solid var(--border-color);
        padding: 20px 10px;
    }
    
    .sidebar-header {
        padding: 15px;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .sidebar-header h3 {
        margin: 0;
        color: var(--text-primary);
        font-size: 18px;
    }
    
    /* Configuration Card */
    .config-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid var(--border-color);
        transition: transform 0.2s ease;
    }
    
    .config-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }
    
    .model-status {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 8px 0;
        padding: 8px;
        border-radius: 8px;
        background: rgba(255,255,255,0.05);
    }
    
    .status-icon {
        font-size: 18px;
        width: 20px;
        text-align: center;
    }
    
    .status-ok {
        color: #10b981;
    }
    
    .status-error {
        color: #ef4444;
    }
    
    .model-name {
        font-size: 14px;
        color: var(--text-primary);
    }
    
    /* Chat Container */
    .chat-container {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid var(--border-color);
        height: 60vh;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
    }
    
    .chat-message {
        display: flex;
        gap: 12px;
        margin-bottom: 15px;
        padding: 12px;
        border-radius: 12px;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    .user-message {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        align-self: flex-end;
        border-left: 4px solid var(--primary);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, rgba(118, 75, 162, 0.2), rgba(102, 126, 234, 0.2));
        align-self: flex-start;
        border-left: 4px solid var(--secondary);
    }
    
    .avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: bold;
        color: white;
    }
    
    .user-avatar {
        background: var(--primary);
    }
    
    .assistant-avatar {
        background: var(--secondary);
    }
    
    .message-content {
        flex: 1;
        line-height: 1.5;
    }
    
    .message-role {
        font-size: 12px;
        opacity: 0.7;
        margin-bottom: 4px;
        font-weight: 600;
    }
    
    /* Input Area */
    .input-container {
        display: flex;
        gap: 10px;
        padding: 10px;
        border-top: 1px solid var(--border-color);
        background: var(--bg-darker);
    }
    
    .input-box {
        flex: 1;
        background: rgba(255,255,255,0.05);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 10px;
        color: var(--text-primary);
        font-size: 14px;
        outline: none;
    }
    
    .send-button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .send-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Model Selector */
    .model-selector {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid var(--border-color);
    }
    
    .model-label {
        font-size: 14px;
        color: var(--text-secondary);
        margin-bottom: 5px;
    }
    
    .model-option {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px;
        border-radius: 6px;
        cursor: pointer;
        transition: background 0.2s ease;
    }
    
    .model-option:hover {
        background: rgba(255,255,255,0.05);
    }
    
    .model-option.selected {
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid var(--primary);
    }
    
    .model-icon {
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        border-radius: 4px;
    }
    
    .model-claude {
        background: #8b5cf6;
    }
    
    .model-gpt {
        background: #10b981;
    }
    
    .model-gemini {
        background: #f59e0b;
    }
    
    .model-ollama {
        background: #ec4899;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 15px;
        color: var(--text-secondary);
        font-size: 12px;
        border-top: 1px solid var(--border-color);
        margin-top: 20px;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Enterprise AI Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MultiModelChatbot()
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = 'ollama'
    
    # Main Header
    st.markdown("""
    <div class="main-header">
        <div class="header-icon">🤖</div>
        <div>
            <h1 class="main-title">Enterprise AI Chatbot</h1>
            <p class="subtitle">Powered by Claude • GPT • Gemini • Ollama</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("""
        <div class="sidebar-header">
            <div style="width: 24px; height: 24px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 14px;">
                ⚙️
            </div>
            <h3>Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Availability
        st.markdown("<h3 style='margin-top: 20px;'>Model Availability</h3>", unsafe_allow_html=True)
        
        # Anthropic (Claude)
        status = "✅" if ANTHROPIC_AVAILABLE else "❌"
        color = "status-ok" if ANTHROPIC_AVAILABLE else "status-error"
        st.markdown(f"""
        <div class="model-status">
            <div class="status-icon {color}">{status}</div>
            <div class="model-name">Anthropic (Claude)</div>
        </div>
        """, unsafe_allow_html=True)
        
        if ANTHROPIC_AVAILABLE:
            claude_key = st.text_input("Claude API Key", type="password", key="claude_key", help="Get from console.anthropic.com")
            if claude_key and st.session_state.chatbot.anthropic_client is None:
                try:
                    st.session_state.chatbot.setup_claude(claude_key)
                    st.success("✅ Claude configured!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # OpenAI (GPT)
        status = "✅" if OPENAI_AVAILABLE else "❌"
        color = "status-ok" if OPENAI_AVAILABLE else "status-error"
        st.markdown(f"""
        <div class="model-status">
            <div class="status-icon {color}">{status}</div>
            <div class="model-name">OpenAI (GPT)</div>
        </div>
        """, unsafe_allow_html=True)
        
        if OPENAI_AVAILABLE:
            openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key", help="Get from platform.openai.com")
            if openai_key and st.session_state.chatbot.openai_client is None:
                try:
                    st.session_state.chatbot.setup_gpt(openai_key)
                    st.success("✅ GPT configured!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Google (Gemini)
        status = "✅" if GEMINI_AVAILABLE else "❌"
        color = "status-ok" if GEMINI_AVAILABLE else "status-error"
        st.markdown(f"""
        <div class="model-status">
            <div class="status-icon {color}">{status}</div>
            <div class="model-name">Google (Gemini)</div>
        </div>
        """, unsafe_allow_html=True)
        
        if GEMINI_AVAILABLE:
            gemini_key = st.text_input("Gemini API Key", type="password", key="gemini_key", help="Get from makersuite.google.com")
            if gemini_key and st.session_state.chatbot.gemini_model is None:
                try:
                    st.session_state.chatbot.setup_gemini(gemini_key)
                    st.success("✅ Gemini configured!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Ollama (Local)
        status = "✅" if OLLAMA_AVAILABLE else "❌"
        color = "status-ok" if OLLAMA_AVAILABLE else "status-error"
        st.markdown(f"""
        <div class="model-status">
            <div class="status-icon {color}">{status}</div>
            <div class="model-name">Ollama (Local)</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 20px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        # Model Selection
        st.markdown("<h3>Active Model</h3>", unsafe_allow_html=True)
        
        available_models = []
        if st.session_state.chatbot.anthropic_client:
            available_models.append('claude')
        if st.session_state.chatbot.openai_client:
            available_models.append('gpt')
        if st.session_state.chatbot.gemini_model:
            available_models.append('gemini')
        available_models.append('ollama')  # Always available
        
        if available_models:
            selected_model = st.selectbox(
                "Select Model",
                available_models,
                format_func=lambda x: {
                    'claude': '🔮 Claude 3.5 Sonnet',
                    'gpt': '🤖 GPT-3.5 Turbo',
                    'gemini': '✨ Gemini Pro',
                    'ollama': '🧠 Ollama (Local)'
                }[x],
                index=available_models.index(st.session_state.current_model) 
                    if st.session_state.current_model in available_models else 0
            )
            st.session_state.current_model = selected_model
        else:
            st.warning("Please configure at least one API key")
        
        # Clear History Button
        if st.button("🗑️ Clear Conversation History", key="clear_history"):
            st.session_state.chatbot.clear_history(st.session_state.current_model)
            st.success(f"Cleared {st.session_state.current_model.upper()} history")
    
    with col2:
        # Chat Interface
        st.markdown(f"""
        <div style="background: var(--card-bg); border-radius: 12px; padding: 15px; border: 1px solid var(--border-color);">
            <h3 style="margin: 0 0 15px 0; color: var(--text-primary);">💬 Chatting with: {st.session_state.current_model.upper()}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat Container
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        
        # Display conversation history
        history = st.session_state.chatbot.get_history(st.session_state.current_model)
        
        for msg in history:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="avatar user-avatar">👤</div>
                    <div class="message-content">
                        <div class="message-role">You</div>
                        {msg['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="avatar assistant-avatar">🤖</div>
                    <div class="message-content">
                        <div class="message-role">AI</div>
                        {msg['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Input Area
        st.markdown("<div class='input-container'>", unsafe_allow_html=True)
        
        # Message input
        user_input = st.text_area(
            "Your message",
            height=50,
            placeholder="Type your message here...",
            label_visibility="collapsed",
            key="user_input"
        )
        
        # Send button
        send_button = st.button("Send", key="send_button", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Handle message submission
        if send_button and user_input.strip():
            # Add user message to chat
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="avatar user-avatar">👤</div>
                <div class="message-content">
                    <div class="message-role">You</div>
                    {user_input}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Get AI response
            with st.spinner("🤖 Thinking..."):
                response = st.session_state.chatbot.chat(
                    st.session_state.current_model,
                    user_input
                )
            
            # Add AI response to chat
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="avatar assistant-avatar">🤖</div>
                <div class="message-content">
                    <div class="message-role">AI</div>
                    {response}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Clear input after sending
            st.rerun()

    # Footer
    st.markdown("""
    <div class="footer">
        Enterprise AI Chatbot • Built with Streamlit • Powered by Multiple AI Models
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
