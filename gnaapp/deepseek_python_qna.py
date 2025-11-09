import streamlit as st
import sqlite3
import json
import random
from typing import List, Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ML Quiz Master",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MLQuizGame:
    def __init__(self, db_path: str = "ml_quiz.db"):
        self.db_path = db_path
        self.init_database()
        self.load_questions()
    
    def init_database(self):
        """Initialize SQLite database for storing questions and user progress"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create questions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                options TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                explanation TEXT,
                category TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create user progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'default',
                category TEXT NOT NULL,
                questions_answered INTEGER DEFAULT 0,
                correct_answers INTEGER DEFAULT 0,
                score INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create session table for storing current quiz state
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quiz_sessions (
                session_id TEXT PRIMARY KEY,
                questions TEXT NOT NULL,
                current_question INTEGER DEFAULT 0,
                score INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_questions(self):
        """Load questions from database"""
        self.questions = self.get_all_questions()
        
        # If no questions in DB, load default questions
        if not self.questions:
            self.load_default_questions()
            self.questions = self.get_all_questions()
    
    def get_all_questions(self) -> List[Dict]:
        """Retrieve all questions from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM questions")
        rows = cursor.fetchall()
        
        questions = []
        for row in rows:
            questions.append({
                'id': row[0],
                'question': row[1],
                'options': json.loads(row[2]),
                'correct_answer': row[3],
                'explanation': row[4],
                'category': row[5],
                'difficulty': row[6]
            })
        
        conn.close()
        return questions
    
    def load_default_questions(self):
        """Load comprehensive set of ML, statistics, and math questions"""
        from questions_data import get_comprehensive_questions
        
        questions_data = get_comprehensive_questions()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for q in questions_data:
            cursor.execute('''
                INSERT INTO questions (question, options, correct_answer, explanation, category, difficulty)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (q['question'], json.dumps(q['options']), q['correct_answer'], 
                  q['explanation'], q['category'], q['difficulty']))
        
        conn.commit()
        conn.close()
        st.success(f"✅ Loaded {len(questions_data)} questions into database!")
    
    def get_questions_by_category(self, category: str = "all") -> List[Dict]:
        """Get questions filtered by category"""
        if category == "all":
            return self.questions
        return [q for q in self.questions if q['category'] == category]
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        categories = list(set(q['category'] for q in self.questions))
        categories.sort()
        return ["all"] + categories
    
    def save_progress(self, category: str, questions_answered: int, correct_answers: int, score: int):
        """Save user progress to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_progress (category, questions_answered, correct_answers, score)
            VALUES (?, ?, ?, ?)
        ''', (category, questions_answered, correct_answers, score))
        
        conn.commit()
        conn.close()
    
    def get_user_progress(self) -> pd.DataFrame:
        """Get user progress as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT category, 
                   SUM(questions_answered) as total_questions,
                   SUM(correct_answers) as total_correct,
                   AVG(CAST(correct_answers as FLOAT) / questions_answered * 100) as accuracy,
                   MAX(timestamp) as last_attempt
            FROM user_progress 
            GROUP BY category
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

def initialize_session_state():
    """Initialize session state variables"""
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'selected_questions' not in st.session_state:
        st.session_state.selected_questions = []
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = []
    if 'quiz_finished' not in st.session_state:
        st.session_state.quiz_finished = False
    if 'category' not in st.session_state:
        st.session_state.category = "all"

def main():
    # Initialize game and session state
    game = MLQuizGame()
    initialize_session_state()
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .category-card {
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin: 0.5rem 0;
        }
        .score-card {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        .question-card {
            padding: 2rem;
            border-radius: 10px;
            border: 2px solid #1f77b4;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
        st.title("🧠 ML Quiz Master")
        st.markdown("---")
        
        menu = st.radio("Navigation", ["🏠 Home", "🎯 Take Quiz", "📊 Progress", "📚 Categories", "ℹ️ About"])
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        progress_df = game.get_user_progress()
        if not progress_df.empty:
            total_questions = progress_df['total_questions'].sum()
            total_correct = progress_df['total_correct'].sum()
            overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
            
            st.metric("Total Questions", int(total_questions))
            st.metric("Correct Answers", int(total_correct))
            st.metric("Overall Accuracy", f"{overall_accuracy:.1f}%")
    
    # Home Page
    if menu == "🏠 Home":
        st.markdown('<h1 class="main-header">🚀 Welcome to ML Quiz Master!</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **📚 Comprehensive Coverage**
            - 1000+ Questions
            - ML Algorithms
            - Statistics & Math
            - Real-world Scenarios
            """)
        
        with col2:
            st.success("""
            **🎯 Learning Features**
            - Multiple Difficulty Levels
            - Detailed Explanations
            - Progress Tracking
            - Performance Analytics
            """)
        
        with col3:
            st.warning("""
            **🏆 Categories**
            - Machine Learning
            - Deep Learning
            - Statistics
            - Mathematics
            - Linear Algebra
            - Calculus
            """)
        
        st.markdown("---")
        
        # Quick start
        st.subheader("🎮 Quick Start Quiz")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            category = st.selectbox("Select Category", game.get_categories(), format_func=lambda x: x.replace('_', ' ').title())
        
        with col2:
            num_questions = st.selectbox("Questions", [5, 10, 15, 20, 25])
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Start Quiz 🚀", use_container_width=True):
                st.session_state.quiz_started = True
                st.session_state.category = category
                st.session_state.current_question = 0
                st.session_state.score = 0
                st.session_state.quiz_finished = False
                
                # Select random questions
                available_questions = game.get_questions_by_category(category)
                st.session_state.selected_questions = random.sample(
                    available_questions, min(num_questions, len(available_questions))
                )
                st.session_state.user_answers = [None] * len(st.session_state.selected_questions)
                st.rerun()
    
    # Take Quiz Page
    elif menu == "🎯 Take Quiz":
        st.title("🎯 ML Quiz Challenge")
        
        if not st.session_state.quiz_started:
            st.info("👆 Start a quiz from the Home page or configure your quiz below!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                category = st.selectbox("Quiz Category", game.get_categories(), 
                                      format_func=lambda x: x.replace('_', ' ').title())
            
            with col2:
                difficulty = st.selectbox("Difficulty Level", ["all", "easy", "medium", "hard"])
            
            with col3:
                num_questions = st.slider("Number of Questions", 5, 50, 10)
            
            if st.button("Start Quiz 🚀", type="primary", use_container_width=True):
                st.session_state.quiz_started = True
                st.session_state.category = category
                st.session_state.current_question = 0
                st.session_state.score = 0
                st.session_state.quiz_finished = False
                
                # Filter questions by difficulty if needed
                available_questions = game.get_questions_by_category(category)
                if difficulty != "all":
                    available_questions = [q for q in available_questions if q['difficulty'] == difficulty]
                
                st.session_state.selected_questions = random.sample(
                    available_questions, min(num_questions, len(available_questions))
                )
                st.session_state.user_answers = [None] * len(st.session_state.selected_questions)
                st.rerun()
        
        else:
            # Display current question
            if st.session_state.current_question < len(st.session_state.selected_questions):
                question_data = st.session_state.selected_questions[st.session_state.current_question]
                
                st.markdown(f'<div class="question-card">', unsafe_allow_html=True)
                
                # Progress bar
                progress = (st.session_state.current_question + 1) / len(st.session_state.selected_questions)
                st.progress(progress)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.subheader(f"Question {st.session_state.current_question + 1}/{len(st.session_state.selected_questions)}")
                with col2:
                    st.info(f"Category: {question_data['category'].replace('_', ' ').title()}")
                with col3:
                    st.warning(f"Difficulty: {question_data['difficulty'].upper()}")
                
                st.markdown(f"### {question_data['question']}")
                
                # Display options
                user_answer = st.radio(
                    "Select your answer:",
                    question_data['options'],
                    key=f"q{st.session_state.current_question}"
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("Submit Answer", type="primary", use_container_width=True):
                        st.session_state.user_answers[st.session_state.current_question] = user_answer
                        
                        if user_answer == question_data['correct_answer']:
                            st.session_state.score += 1
                            st.success("✅ Correct!")
                        else:
                            st.error("❌ Incorrect!")
                        
                        # Show explanation
                        with st.expander("📖 Explanation"):
                            st.write(question_data['explanation'])
                            st.write(f"**Correct Answer:** {question_data['correct_answer']}")
                        
                        if st.button("Next Question ➡️", use_container_width=True):
                            st.session_state.current_question += 1
                            st.rerun()
                
                with col2:
                    st.markdown(f"### Current Score: {st.session_state.score}/{st.session_state.current_question + 1}")
            
            else:
                # Quiz finished
                st.session_state.quiz_finished = True
                st.balloons()
                
                st.markdown('<div class="score-card">', unsafe_allow_html=True)
                st.markdown(f"# 🎉 Quiz Completed!")
                st.markdown(f"## Final Score: {st.session_state.score}/{len(st.session_state.selected_questions)}")
                accuracy = (st.session_state.score / len(st.session_state.selected_questions)) * 100
                st.markdown(f"## Accuracy: {accuracy:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Save progress
                game.save_progress(
                    st.session_state.category,
                    len(st.session_state.selected_questions),
                    st.session_state.score,
                    st.session_state.score
                )
                
                # Detailed results
                st.subheader("📊 Detailed Results")
                for i, (question, user_answer) in enumerate(zip(st.session_state.selected_questions, st.session_state.user_answers)):
                    with st.expander(f"Question {i+1}: {question['question'][:50]}..."):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Your Answer:** {user_answer}")
                            st.write(f"**Correct Answer:** {question['correct_answer']}")
                            if user_answer == question['correct_answer']:
                                st.success("✅ Correct")
                            else:
                                st.error("❌ Incorrect")
                        with col2:
                            st.write(f"**Explanation:** {question['explanation']}")
                
                if st.button("🔄 Take Another Quiz", use_container_width=True):
                    st.session_state.quiz_started = False
                    st.session_state.quiz_finished = False
                    st.rerun()
    
    # Progress Page
    elif menu == "📊 Progress":
        st.title("📊 Your Learning Progress")
        
        progress_df = game.get_user_progress()
        
        if progress_df.empty:
            st.info("No quiz attempts yet. Start learning! 🚀")
        else:
            # Overall statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_questions = progress_df['total_questions'].sum()
            total_correct = progress_df['total_correct'].sum()
            overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
            
            with col1:
                st.metric("Total Questions", int(total_questions))
            with col2:
                st.metric("Correct Answers", int(total_correct))
            with col3:
                st.metric("Overall Accuracy", f"{overall_accuracy:.1f}%")
            with col4:
                st.metric("Categories Attempted", len(progress_df))
            
            # Accuracy by category chart
            st.subheader("📈 Accuracy by Category")
            fig = px.bar(progress_df, x='category', y='accuracy', 
                        color='accuracy', color_continuous_scale='Viridis',
                        title="Accuracy Percentage by Category")
            st.plotly_chart(fig, use_container_width=True)
            
            # Questions by category
            st.subheader("📊 Questions by Category")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(progress_df, values='total_questions', names='category',
                               title="Questions Distribution by Category")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Progress table
                st.dataframe(progress_df, use_container_width=True)
    
    # Categories Page
    elif menu == "📚 Categories":
        st.title("📚 Learning Categories")
        
        categories = game.get_categories()
        questions_by_category = {
            category: len(game.get_questions_by_category(category))
            for category in categories
        }
        
        # Display category cards
        cols = st.columns(3)
        for idx, (category, count) in enumerate(questions_by_category.items()):
            with cols[idx % 3]:
                st.markdown(f'''
                <div class="category-card">
                    <h3>{category.replace('_', ' ').title()}</h3>
                    <p><strong>{count}</strong> questions available</p>
                </div>
                ''', unsafe_allow_html=True)
                
                if st.button(f"Start {category.title()} Quiz", key=category, use_container_width=True):
                    st.session_state.quiz_started = True
                    st.session_state.category = category
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                    st.session_state.quiz_finished = False
                    
                    available_questions = game.get_questions_by_category(category)
                    st.session_state.selected_questions = random.sample(available_questions, min(10, len(available_questions)))
                    st.session_state.user_answers = [None] * len(st.session_state.selected_questions)
                    st.switch_page("🎯 Take Quiz")
    
    # About Page
    elif menu == "ℹ️ About":
        st.title("ℹ️ About ML Quiz Master")
        
        st.markdown("""
        ## 🧠 ML Quiz Master
        
        A comprehensive interactive learning platform for mastering Machine Learning, 
        Statistics, and Mathematics concepts through engaging quizzes.
        
        ### 🎯 Features
        
        - **1000+ Carefully Crafted Questions** covering all major ML topics
        - **Multiple Difficulty Levels** from beginner to advanced
        - **Detailed Explanations** for deeper understanding
        - **Progress Tracking** to monitor your learning journey
        - **Interactive Visualizations** of your performance
        
        ### 📚 Categories Covered
        
        1. **Machine Learning Algorithms**
           - Linear Regression, Logistic Regression
           - Decision Trees, Random Forest
           - SVM, K-Nearest Neighbors
           - Clustering Algorithms
        
        2. **Deep Learning**
           - Neural Networks
           - CNN, RNN, LSTM
           - Transformers
           - Optimization Techniques
        
        3. **Statistics & Probability**
           - Descriptive Statistics
           - Inferential Statistics
           - Probability Distributions
           - Hypothesis Testing
        
        4. **Mathematics**
           - Linear Algebra
           - Calculus
           - Optimization
           - Information Theory
        
        ### 🚀 Getting Started
        
        1. Choose a category from the Home page
        2. Select your preferred quiz settings
        3. Start answering questions and learning!
        4. Track your progress in the Progress section
        
        Happy Learning! 🎓
        """)

if __name__ == "__main__":
    main()