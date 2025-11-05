import streamlit as st
import os
from typing import Dict, List

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


class MultiModelChatbot:
    """
    A chatbot that can interact with multiple AI models:
    - Claude (Anthropic)
    - GPT (OpenAI)
    - Gemini (Google)
    """
    
    def __init__(self):
        self.conversation_history: Dict[str, List[Dict]] = {
            'claude': [],
            'gpt': [],
            'gemini': []
        }
        
        self.anthropic_client = None
        self.openai_client = None
        self.gemini_model = None
        
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
        
    def chat_with_claude(self, message: str, model: str = "claude-3-5-sonnet-20241022") -> str:
        """Send message to Claude and get response"""
        if not self.anthropic_client:
            return "Error: Claude API not configured"
        
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
            return f"Error: {str(e)}"
    
    def chat_with_gpt(self, message: str, model: str = "gpt-4") -> str:
        """Send message to GPT and get response"""
        if not self.openai_client:
            return "Error: OpenAI API not configured"
        
        self.conversation_history['gpt'].append({
            "role": "user",
            "content": message
        })
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=self.conversation_history['gpt']
            )
            
            assistant_message = response.choices[0].message.content
            self.conversation_history['gpt'].append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat_with_gemini(self, message: str) -> str:
        """Send message to Gemini and get response"""
        if not self.gemini_model:
            return "Error: Gemini API not configured"
        
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
            return f"Error: {str(e)}"
    
    def chat(self, model_name: str, message: str) -> str:
        """Universal chat method"""
        if model_name.lower() == 'claude':
            return self.chat_with_claude(message)
        elif model_name.lower() == 'gpt':
            return self.chat_with_gpt(message)
        elif model_name.lower() == 'gemini':
            return self.chat_with_gemini(message)
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


def main():
    st.set_page_config(
        page_title="Multi-Model AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Multi-Model AI Chatbot")
    st.markdown("Chat with Claude, GPT, and Gemini in one place!")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MultiModelChatbot()
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = 'claude'
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model availability status
        st.subheader("Package Status")
        st.write(f"{'✅' if ANTHROPIC_AVAILABLE else '❌'} Anthropic")
        st.write(f"{'✅' if OPENAI_AVAILABLE else '❌'} OpenAI")
        st.write(f"{'✅' if GEMINI_AVAILABLE else '❌'} Google AI")
        
        st.divider()
        
        # API Keys
        st.subheader("API Keys")
        
        if ANTHROPIC_AVAILABLE:
            claude_key = st.text_input("Claude API Key", type="password", key="claude_key")
            if claude_key and st.session_state.chatbot.anthropic_client is None:
                try:
                    st.session_state.chatbot.setup_claude(claude_key)
                    st.success("Claude configured!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if OPENAI_AVAILABLE:
            openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
            if openai_key and st.session_state.chatbot.openai_client is None:
                try:
                    st.session_state.chatbot.setup_gpt(openai_key)
                    st.success("GPT configured!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if GEMINI_AVAILABLE:
            gemini_key = st.text_input("Gemini API Key", type="password", key="gemini_key")
            if gemini_key and st.session_state.chatbot.gemini_model is None:
                try:
                    st.session_state.chatbot.setup_gemini(gemini_key)
                    st.success("Gemini configured!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.divider()
        
        # Model selection
        st.subheader("Select Model")
        available_models = []
        if st.session_state.chatbot.anthropic_client:
            available_models.append('claude')
        if st.session_state.chatbot.openai_client:
            available_models.append('gpt')
        if st.session_state.chatbot.gemini_model:
            available_models.append('gemini')
        
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
    
    # Main chat interface
    st.subheader(f"💬 Chatting with: {st.session_state.current_model.upper()}")
    
    # Display chat history
    history = st.session_state.chatbot.get_history(st.session_state.current_model)
    
    chat_container = st.container()
    with chat_container:
        for msg in history:
            if msg['role'] == 'user':
                st.chat_message("user").write(msg['content'])
            else:
                st.chat_message("assistant").write(msg['content'])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Display user message
        st.chat_message("user").write(user_input)
        
        # Get AI response
        with st.spinner(f"Waiting for {st.session_state.current_model.upper()}..."):
            response = st.session_state.chatbot.chat(
                st.session_state.current_model,
                user_input
            )
        
        # Display assistant response
        st.chat_message("assistant").write(response)
        
        # Rerun to update chat display
        st.rerun()


if __name__ == "__main__":
    main()
