"""
Streamlit UI for the Interactive Coding Tutor.
"""

import streamlit as st
import logging
import sys
import os
import yaml
from typing import Dict, List, Any
import traceback

# Add src to path for imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, '..')
root_dir = os.path.join(src_dir, '..')
sys.path.append(src_dir)

try:
    from business_logic.core import (
        handle_query, explain_concept, generate_exercise, evaluate_user_code,
        get_supported_languages, get_topics, validate_inputs, create_chat_message,
        TutorError
    )
    from helpers.lm_studio import get_available_models, test_connection, LMStudioError
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIG = {
    "llm_endpoint": "http://localhost:1234",
    "supported_languages": ["python", "javascript", "java", "cpp", "csharp"],
    "default_model": "phi-3-mini"
}

# Page configuration
st.set_page_config(
    page_title="Interactive Coding Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(ttl=30)
def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_path = os.path.join(root_dir, "config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        return DEFAULT_CONFIG


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    defaults = {
        "chat_history": [],
        "selected_language": "python",
        "selected_model": None,
        "current_exercise": None,
        "current_explanation": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(ttl=30)
def check_lm_studio_connection(endpoint: str, config: Dict[str, Any]) -> bool:
    """Check if LM Studio is accessible"""
    timeout = config.get("app_settings", {}).get("connection_check_interval", 5)
    return test_connection(endpoint=endpoint, timeout=timeout)


@st.cache_data(ttl=60)
def get_models(endpoint: str, config: Dict[str, Any]) -> List[str]:
    """Get available models from LM Studio"""
    try:
        timeout = config.get("generation_settings", {}).get("timeout", 10)
        return get_available_models(endpoint=endpoint, timeout=timeout)
    except LMStudioError:
        return ["No models available"]


def render_sidebar(config: Dict[str, Any]) -> bool:
    """Render the sidebar with model and language selection"""
    with st.sidebar:
        st.title("🎓 Coding Tutor")
        st.markdown("---")
        
        endpoint = config.get("llm_endpoint", "http://localhost:1234")
        
        # Connection status
        if not _render_connection_status(endpoint, config):
            return False
        
        # Model selection
        if not _render_model_selection(endpoint, config):
            return False
        
        # Language selection
        _render_language_selection()
        
        # Action buttons
        _render_sidebar_buttons()
        
        return True


def _render_connection_status(endpoint: str, config: Dict[str, Any]) -> bool:
    """Render LM Studio connection status"""
    is_connected = check_lm_studio_connection(endpoint, config)
    
    if is_connected:
        st.success("✅ LM Studio Connected")
        return True
    else:
        st.error("❌ LM Studio Disconnected")
        st.warning("Please start LM Studio and load a model")
        return False


def _render_model_selection(endpoint: str, config: Dict[str, Any]) -> bool:
    """Render AI model selection"""
    st.subheader("🤖 AI Model")
    available_models = get_models(endpoint, config)
    
    if not available_models or available_models[0] == "No chat models available":
        st.error("No chat models available in LM Studio")
        st.warning("Make sure to load a chat/instruct model")
        return False
    
    model_index = 0
    if st.session_state.selected_model in available_models:
        model_index = available_models.index(st.session_state.selected_model)
    
    st.session_state.selected_model = st.selectbox(
        "Choose a model:",
        available_models,
        index=model_index,
        help="Select the AI model to use for tutoring"
    )
    
    if len(available_models) < 3:
        st.info("ℹ️ Embedding models filtered out")
    
    return True


def _render_language_selection():
    """Render programming language selection"""
    st.subheader("💻 Programming Language")
    languages = get_supported_languages()
    lang_codes = list(languages.keys())
    
    lang_index = 0
    if st.session_state.selected_language in lang_codes:
        lang_index = lang_codes.index(st.session_state.selected_language)
    
    st.session_state.selected_language = st.selectbox(
        "Choose a language:",
        lang_codes,
        format_func=lambda x: languages[x],
        index=lang_index,
        help="Select the programming language to learn"
    )


def _render_sidebar_buttons():
    """Render sidebar action buttons"""
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.current_exercise = None
        st.session_state.current_explanation = None
        st.rerun()
        
    if st.button("🔄 Reload Config", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
def handle_chat_input(user_input: str, config: Dict[str, Any]) -> str:
    """Handle new chat input"""
    try:
        validate_inputs(user_input, st.session_state.selected_language, st.session_state.selected_model)
        
        endpoint = config.get("llm_endpoint", "http://localhost:1234")
        
        return handle_query(
            user_input=user_input,
            language=st.session_state.selected_language,
            model=st.session_state.selected_model,
            history=st.session_state.chat_history[-10:],
            endpoint=endpoint,
            config=config
        )
        
    except (TutorError, LMStudioError) as e:
        return f"Sorry, I encountered an error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        return f"An unexpected error occurred: {e}"


def _display_chat_history():
    """Display chat history messages"""
    for message in st.session_state.chat_history:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        with st.chat_message(role):
            if role == "user":
                st.write(content)
            else:
                st.markdown(content)


def _process_user_message(user_input: str):
    """Process new user message and generate response"""
    # Add user message
    user_msg = create_chat_message("user", user_input)
    st.session_state.chat_history.append(user_msg)
    
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = load_config()
            response = handle_chat_input(user_input, config)
            st.markdown(response)
            
            # Add assistant response
            assistant_msg = create_chat_message("assistant", response)
            st.session_state.chat_history.append(assistant_msg)


def render_chat_interface():
    """Render the main chat interface"""
    language_name = get_supported_languages()[st.session_state.selected_language]
    st.markdown(f"<h2 style='text-align: center;'>💬 Chat with your {language_name} tutor</h2>", 
                unsafe_allow_html=True)
    
    # Display chat history
    _display_chat_history()
    
    # Chat input
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_input = st.chat_input("Ask me anything about programming...")
    
    if user_input:
        _process_user_message(user_input)


def _generate_explanation(selected_topic: str) -> None:
    """Generate explanation for selected topic"""
    col1_spinner, col2_spinner, col3_spinner = st.columns([1, 1, 1])
    with col2_spinner:
        with st.spinner("Generating..."):
            try:
                config = load_config()
                endpoint = config.get("llm_endpoint", "http://localhost:1234")
                
                explanation = explain_concept(
                    concept=selected_topic.replace('_', ' '),
                    language=st.session_state.selected_language,
                    model=st.session_state.selected_model,
                    endpoint=endpoint,
                    config=config
                )
                st.session_state.current_explanation = explanation
                
            except (TutorError, LMStudioError) as e:
                st.error(f"Failed to generate explanation: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                logger.error(f"Unexpected error in concept explainer: {e}")


def _display_explanation():
    """Display current explanation with clear button"""
    if st.session_state.get('current_explanation'):
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>Current Explanation:</h3>", unsafe_allow_html=True)
        st.markdown(st.session_state.current_explanation)
        
        col1_clear, col2_clear, col3_clear = st.columns([1, 1, 1])
        with col2_clear:
            if st.button("🗑️ Clear Explanation", use_container_width=True):
                st.session_state.current_explanation = None
                st.rerun()


def render_concept_explainer():
    """Render the concept explanation interface"""
    st.markdown("<h2 style='text-align: center;'>📖 Concept Explainer</h2>", unsafe_allow_html=True)
    
    topics = get_topics(st.session_state.selected_language)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        selected_topic = st.selectbox(
            "Choose a concept to learn about:",
            topics,
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Select a programming concept for detailed explanation"
        )
        
        explain_button = st.button("📚 Explain Concept", use_container_width=True)
        
        if explain_button and selected_topic:
            _generate_explanation(selected_topic)
    
    _display_explanation()


def _generate_exercise(selected_topic: str) -> None:
    """Generate exercise for selected topic"""
    col1_spinner, col2_spinner, col3_spinner = st.columns([1, 1, 1])
    with col2_spinner:
        with st.spinner("Generating..."):
            try:
                config = load_config()
                endpoint = config.get("llm_endpoint", "http://localhost:1234")
                
                exercise = generate_exercise(
                    topic=selected_topic,
                    language=st.session_state.selected_language,
                    model=st.session_state.selected_model,
                    endpoint=endpoint,
                    config=config
                )
                st.session_state.current_exercise = exercise
                
            except (TutorError, LMStudioError) as e:
                st.error(f"Failed to generate exercise: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                logger.error(f"Unexpected error in exercise generator: {e}")


def _evaluate_code(user_code: str) -> None:
    """Evaluate user submitted code"""
    with st.spinner("Evaluating your code..."):
        try:
            config = load_config()
            endpoint = config.get("llm_endpoint", "http://localhost:1234")
            
            feedback = evaluate_user_code(
                code=user_code,
                task=st.session_state.current_exercise["task_description"],
                language=st.session_state.selected_language,
                model=st.session_state.selected_model,
                endpoint=endpoint,
                config=config
            )
            
            st.markdown("<h3 style='text-align: center;'>Feedback:</h3>", unsafe_allow_html=True)
            st.markdown(feedback)
            
        except (TutorError, LMStudioError) as e:
            st.error(f"Failed to evaluate code: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.error(f"Unexpected error in code evaluation: {e}")


def _display_exercise():
    """Display current exercise with code input and evaluation"""
    if not st.session_state.current_exercise:
        return
        
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>Current Exercise:</h3>", unsafe_allow_html=True)
    st.markdown(st.session_state.current_exercise["exercise_text"])
    
    st.markdown("<h3 style='text-align: center;'>Your Solution:</h3>", unsafe_allow_html=True)
    language_name = get_supported_languages()[st.session_state.selected_language]
    user_code = st.text_area(
        "Write your code here:",
        height=200,
        placeholder=f"# Write your {language_name} code here...",
        help="Enter your solution to the exercise above"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        submit_button = st.button("✅ Submit Code", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear Exercise", use_container_width=True)
    
    if submit_button and user_code.strip():
        _evaluate_code(user_code)
    elif submit_button:
        st.warning("Please enter some code before submitting.")
    
    if clear_button:
        st.session_state.current_exercise = None
        st.rerun()


def render_exercise_generator():
    """Render the exercise generation and evaluation interface"""
    st.markdown("<h2 style='text-align: center;'>🏋️ Practice Exercises</h2>", unsafe_allow_html=True)
    
    topics = get_topics(st.session_state.selected_language)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        selected_topic = st.selectbox(
            "Choose a topic for practice:",
            topics,
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Select a topic to generate a practice exercise"
        )
        
        generate_button = st.button("🎯 Generate Exercise", use_container_width=True)
    
    if generate_button and selected_topic:
        _generate_exercise(selected_topic)
    
    _display_exercise()


def _apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            width: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def _render_footer():
    """Render the application footer"""
    st.markdown(
        """
        <div style='
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: var(--background-color);
            text-align: right;
            padding: 5px 15px;
            border-top: 1px solid var(--secondary-background-color);
            z-index: 999;
            font-size: 0.75em;
            opacity: 0.8;
        '>
        Developed by tpravinos | Built with ❤️ using Streamlit and LM Studio
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    """Main application function"""
    initialize_session_state()
    config = load_config()
    
    # Render sidebar
    if not render_sidebar(config):
        st.error("Please ensure LM Studio is running with models loaded.")
        st.stop()
    
    # Apply styling
    _apply_custom_css()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📖 Learn Concepts", "🏋️ Practice"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_concept_explainer()
    
    with tab3:
        render_exercise_generator()
    
    # Render footer
    _render_footer()


if __name__ == "__main__":
    main()
