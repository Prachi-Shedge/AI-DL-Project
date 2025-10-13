import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# LangChain import for creating the tool
from langchain_core.tools import tool

# --- Import all required specialized modules ---
try:
    from nodes.llm_question_classifier import LLMQuestionClassifier, QuestionIntent
    from nodes.hybrid_calculation_engine import HybridCalculationEngine
    from gemma_llm import GemmaLLM
    from groq_llm import GroqLLM
except ImportError:
    print("FATAL ERROR: Could not import required modules. Please ensure all tool files are in the correct path.")
    # Provide minimal fallbacks for basic script loading
    class LLMQuestionClassifier:
        def analyze_question_intent(self, q, df_context): return type('obj', (object,), {'intent': 'descriptive', 'confidence': 0, 'reasoning': 'Fallback'})()
    class HybridCalculationEngine:
        def analyze_with_calculations(self, q, df): return "Hybrid Engine not available."
    class GemmaLLM:
        def __init__(self, **kwargs): pass
        def is_available(self): return False
        def __call__(self, prompt): return "Gemma LLM not available."
    class QuestionIntent:
        STATISTICAL, COMPARATIVE, ANALYTICAL, DESCRIPTIVE = "statistical", "comparative", "analytical", "descriptive"

# Optional Plotly imports for visualization
try:
    import plotly.express as px
    import plotly.io as pio
except ImportError:
    px, pio = None, None
    print("Warning: Plotly not found. Visualization will be disabled.")

# ================== THE LANGCHAIN TOOL ==================
@tool
def analyze_user_question(question: str, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Answers a user's question about the dataset by using an LLM to classify the question's
    intent and then routing it to the appropriate specialized analysis engine for calculation,
    visualization, or descriptive answers. This is the primary tool for all data analysis queries.
    """
    print(f"🔬 Tool 'analyze_user_question' starting for question: '{question}'")
    try:
        engine = UnifiedAnalysisEngine()
        result = engine.analyze_question(question, df, dataset_name)
        print("✅ Tool 'analyze_user_question' finished successfully.")
        return result
    except Exception as e:
        print(f"❌ Error in 'analyze_user_question' tool: {e}")
        return {"error": str(e), "answer": f"An error occurred during analysis: {e}"}

# ========== INTERNAL CLASSES AND LOGIC ==========

class InsightAgent:
    """
    Specialized agent for generating descriptive text answers and LLM-powered visualizations.
    """
    def __init__(self):
        self.desc_llm = GemmaLLM(model_name="google/gemma-2-9b-it", temperature=0.2, max_tokens=1500)
        self.viz_llm = GroqLLM(model_name="openai/gpt-oss-120b", temperature=0.0, max_tokens=1000)

    def generate_descriptive_answer(self, question: str, df: pd.DataFrame, dataset_name: str) -> str:
        if not self.desc_llm.is_available(): return "Insight LLM is not available."
        context = (f"Dataset: '{dataset_name}'\nShape: {df.shape}\nColumns: {df.columns.tolist()}\nHead:\n{df.head(3).to_markdown()}")
        prompt = (f"Context:\n{context}\n\nUser Question: \"{question}\"\n\nAnswer the question based only on the context provided.")
        return self.desc_llm(prompt)

    def create_visualization(self, df: pd.DataFrame, question: str):
        """
        Dynamically generates a Plotly visualization using an LLM to write the code.
        This version includes robust checks for incomplete or invalid code generation.
        """
        if not px or not self.viz_llm.is_available(): return None, None

        prompt = self._create_viz_prompt(df, question)
        viz_code = self.viz_llm(prompt)
        if not isinstance(viz_code, str):
            viz_code = getattr(viz_code, 'content', '') or str(viz_code)

        cleaned_code = self._clean_generated_code(viz_code)

        # --- THIS IS THE FIX ---
        # Add a more robust check to ensure the LLM returned a full function call, not just a function name.
        if not cleaned_code or '(' not in cleaned_code or ')' not in cleaned_code:
            print(f"❌ LLM returned incomplete code: '{cleaned_code}'. A full function call is required.")
            return None, None
            
        print(f"✅ Generated Plotly Code for '{question}': `{cleaned_code}`")

        try:
            scope = {"df": df, "px": px, "pd": pd, "np": np}
            fig = eval(cleaned_code, scope)
            
            if fig and hasattr(fig, 'to_html'):
                return fig.to_html(full_html=False, include_plotlyjs='cdn'), pio.to_json(fig)
            else:
                print("❌ Executed code did not produce a valid Plotly figure.")

        except Exception as e:
            print(f"❌ Visualization execution failed for '{question}': {e}\n   Code: {cleaned_code}")
        return None, None

    def _create_viz_prompt(self, df: pd.DataFrame, question: str) -> str:
        col_details = "\n".join([f"- '{col}' (type: {dtype})" for col, dtype in df.dtypes.items()])
        return f"""
        Given a pandas DataFrame named 'df', write a single, executable line of Python code using the Plotly Express library (aliased as `px`) to best visualize the answer to the user's question.
        USER QUESTION: "{question}"
        DATAFRAME COLUMNS:\n{col_details}
        AVAILABLE PLOT TYPES: px.histogram, px.bar, px.scatter, px.box, px.pie, px.line
        RULES:
        1.  Choose the MOST APPROPRIATE plot type from the list.
        2.  Your output must be a single line of Python code that returns a Plotly figure object.
        3.  Do not include `fig =`, explanations, comments, or markdown.
        CODE:
        """.strip()

    def _clean_generated_code(self, code: str) -> str:
        code = str(code).strip().replace("`python", "").replace("`", "")
        return code[6:] if code.startswith("fig = ") else code

class UnifiedAnalysisEngine:
    # (This class remains unchanged from the previous correct version)
    def __init__(self):
        self.classifier = LLMQuestionClassifier()
        self.hybrid_engine = HybridCalculationEngine()
        self.insight_agent = InsightAgent()

    def analyze_question(self, question: str, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        df_context = f"Dataset has columns: {df.columns.tolist()}"
        classification = self.classifier.analyze_question_intent(question, df_context)
        print(f"🎯 LLM Classification: {classification.intent.value.upper()} (Confidence: {classification.confidence:.1%})")
        print(f"   Reasoning: {classification.reasoning}")
        intent = classification.intent
        if "eda" in question.lower() or intent == QuestionIntent.EXPLORATORY:
            return self._handle_eda_question(question, df, classification)
        elif intent in [QuestionIntent.STATISTICAL, QuestionIntent.COMPARATIVE, QuestionIntent.ANALYTICAL]:
            return self._handle_statistical_question(question, df, classification)
        else:
            return self._handle_descriptive_question(question, df, dataset_name, classification)

    def _handle_statistical_question(self, question: str, df: pd.DataFrame, classification) -> Dict[str, Any]:
        print(f"⚡ Using Hybrid Calculation Engine for: {classification.intent.value}")
        calculation_result = self.hybrid_engine.analyze_with_calculations(question, df)
        viz_html, viz_json = self.insight_agent.create_visualization(df, question)
        return {
            'question': question, 'answer': calculation_result, 'visualization_html': viz_html,
            'visualization_json': viz_json, 'method': f'LLM-Classified ({classification.intent.value}) -> Hybrid Engine'
        }

    def _handle_eda_question(self, question: str, df: pd.DataFrame, classification) -> Dict[str, Any]:
        print("🧭 Performing comprehensive Exploratory Data Analysis (EDA)...")
        eda_script_question = "Perform a comprehensive EDA. Provide dataset shape, missing values, descriptive statistics for numeric columns, and the top 5 absolute correlations between numeric variables."
        summary_result = self.hybrid_engine.analyze_with_calculations(eda_script_question, df)
        viz_questions = self._generate_eda_viz_questions(df)
        print(f"✅ EDA summary complete. Generating {len(viz_questions)} targeted visualizations...")
        all_viz_html = ""
        first_viz_json = None
        for i, viz_q in enumerate(viz_questions):
            html, json_data = self.insight_agent.create_visualization(df, viz_q)
            if html:
                all_viz_html += f"<div><h2>{viz_q}</h2>{html}</div><hr>"
                if i == 0: first_viz_json = json_data
        return {
            'question': question, 'answer': summary_result, 'visualization_html': all_viz_html,
            'visualization_json': first_viz_json, 'method': f'LLM-Classified (EDA) -> Hybrid Engine & Multi-Plot Insights'
        }

    def _generate_eda_viz_questions(self, df: pd.DataFrame) -> list:
        questions = []
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in numeric_cols[:2]: questions.append(f"Plot the distribution of {col}")
        for col in categorical_cols[:2]:
            if df[col].nunique() < 20: questions.append(f"Show the counts for each category in {col}")
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr().abs()
                np.fill_diagonal(corr_matrix.values, 0)
                col1, col2 = corr_matrix.unstack().idxmax()
                questions.append(f"Explore the relationship between {col1} and {col2}")
            except Exception:
                questions.append(f"Explore the relationship between {numeric_cols[0]} and {numeric_cols[1]}")
        return questions

    def _handle_descriptive_question(self, question: str, df: pd.DataFrame, dataset_name: str, classification) -> Dict[str, Any]:
        print("📋 Routing Descriptive Question...")
        q_lower = question.lower()
        if 'columns' in q_lower or 'variables' in q_lower:
            answer = f"The dataset '{dataset_name}' contains {len(df.columns)} columns: {', '.join(df.columns.tolist())}"
            method = 'descriptive_summary'
        elif 'shape' in q_lower or 'size' in q_lower:
            answer = f"The dataset has {df.shape[0]:,} rows and {df.shape[1]} columns."
            method = 'descriptive_summary'
        else:
            print("🧠 Rerouting to InsightAgent for a qualitative answer...")
            answer = self.insight_agent.generate_descriptive_answer(question, df, dataset_name)
            method = f'LLM-Classified ({classification.intent.value}) -> Insight Agent'
        return {
            'question': question, 'answer': answer, 'visualization_html': None,
            'visualization_json': None, 'method': method
        }
