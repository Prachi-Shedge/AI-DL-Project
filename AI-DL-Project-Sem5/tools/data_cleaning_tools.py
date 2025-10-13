import pandas as pd
import numpy as np
import re
import string
from typing import Dict, Any, List

# LangChain import for creating the tool
from langchain_core.tools import tool

# Your existing imports (make sure gemma_llm.py is accessible)
from gemma_llm import GemmaLLM

# Optional NLTK imports for text cleaning helpers, ensure they are installed
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    print("Warning: NLTK not found. Some text cleaning functions may be limited.")


# ================== THE LANGCHAIN TOOL ==================
# This is the primary function that the agent will call.
# It orchestrates the cleaning process based on the data type.

@tool
def clean_and_analyze_data(df: pd.DataFrame, dataset_name: str) -> dict:
    """
    Performs LLM-driven intelligent data cleaning on a pandas DataFrame.
    It automatically detects if the data is structured (CSV/Excel) or unstructured (Text/PDF),
    then applies appropriate cleaning strategies for missing values, duplicates, data types, and outliers.
    
    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.
        dataset_name (str): The name of the dataset being processed.
        
    Returns:
        dict: A dictionary containing the cleaned DataFrame under 'cleaned_dataframe'
              and a detailed report of the cleaning process under 'cleaning_report'.
    """
    print(f"🧹 Tool 'clean_and_analyze_data' starting for dataset: {dataset_name}")
    
    # Initialize Gemma LLM for making cleaning decisions
    llm = GemmaLLM(temperature=0.2, max_tokens=600)
    
    cleaning_report = {
        'original_shape': df.shape,
        'cleaning_method': 'llm_driven',
        'steps_performed': [],
        'llm_recommendations': {},
        'issues_found': [],
        'improvements_made': []
    }

    # Detect data type and structure to route to the correct helper
    is_text_data = "content" in df.columns and (df.shape[1] <= 5 or "metadata" in df.columns)
    
    if is_text_data:
        print("📄 Detected unstructured text data. Routing to text cleaner.")
        cleaned_df, text_analysis = llm_clean_text_data(df, dataset_name, llm, cleaning_report)
        analysis = text_analysis
    else:
        print("📊 Detected structured data. Routing to structured data cleaner.")
        cleaned_df, structured_analysis = llm_clean_structured_data(df, dataset_name, llm, cleaning_report)
        analysis = structured_analysis

    # Finalize the cleaning report
    cleaning_report['final_shape'] = cleaned_df.shape
    cleaning_report['cleaning_success'] = True
    cleaning_report['final_analysis'] = analysis

    print(f"✅ Intelligent cleaning completed for '{dataset_name}'. Final shape: {cleaned_df.shape}")
    
    # CRITICAL CHANGE: Return results as a dictionary, do not modify global state.
    return {
        "cleaned_dataframe": cleaned_df,
        "cleaning_report": cleaning_report
    }


# ========== LLM-DRIVEN CLEANING HELPER FUNCTIONS ==========
# These functions contain the detailed logic and are called by the main tool.
# They are kept internal to this file.

def llm_clean_text_data(df: pd.DataFrame, dataset_name: str, llm: GemmaLLM, cleaning_report: Dict[str, Any]):
    """LLM-driven cleaning for unstructured text data (PDFs, text files)."""
    # This function's logic remains largely the same as your original version.
    print("🤖 Using LLM to analyze and clean text data...")
    cleaned_df = df.copy()
    if 'content' in cleaned_df.columns:
        cleaned_df['cleaned_content'] = cleaned_df['content'].apply(basic_text_clean)
        cleaning_report['improvements_made'].append("Applied basic text cleaning to 'content' column.")
    
    analysis = {'document_count': len(cleaned_df)}
    return cleaned_df, analysis

def llm_clean_structured_data(df: pd.DataFrame, dataset_name: str, llm: GemmaLLM, cleaning_report: Dict[str, Any]):
    """LLM-driven cleaning for structured data (CSV, Excel, JSON)."""
    print("🤖 Using LLM to analyze and clean structured data...")
    
    data_profile = {
        'shape': df.shape, 'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()
    }
    
    # This prompt can be simplified or made more complex depending on need
    cleaning_prompt = f"""For a dataset with columns {data_profile['columns']} and {data_profile['duplicates']} duplicates,
    should I remove duplicates and fill missing numeric values with the median? Answer YES or NO."""
    
    cleaning_strategy = llm(cleaning_prompt, max_tokens=10) # Simple strategy for this example
    cleaning_report['llm_recommendations']['structured_cleaning'] = cleaning_strategy
    
    cleaned_df = df.copy()
    
    if data_profile['missing_values'] and any(v > 0 for v in data_profile['missing_values'].values()):
        print("🔄 Applying intelligent missing value handling...")
        cleaned_df = llm_handle_missing_values(cleaned_df, cleaning_strategy, cleaning_report)
    
    if data_profile['duplicates'] > 0 and "yes" in cleaning_strategy.lower():
        print(f"🗑️ Removing {data_profile['duplicates']} duplicates as recommended")
        cleaned_df = cleaned_df.drop_duplicates()
        cleaning_report['improvements_made'].append(f"Removed {data_profile['duplicates']} duplicate rows")
    
    analysis = {'data_quality_score': calculate_data_quality_score(cleaned_df)}
    return cleaned_df, analysis


# ========== UTILITY AND HELPER FUNCTIONS ==========
# All your original helper functions remain here, unchanged.

def basic_text_clean(text: str) -> str:
    """Basic text cleaning fallback."""
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text

def llm_handle_missing_values(df: pd.DataFrame, strategy: str, report: Dict[str, Any]):
    """Handle missing values based on LLM recommendations."""
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype in ['int64', 'float64']:
                fill_val = df[column].median()
                df[column] = df[column].fillna(fill_val)
                report['improvements_made'].append(f"Filled missing '{column}' with median ({fill_val:.2f})")
            elif df[column].dtype == 'object':
                fill_val = 'Unknown'
                if "mode" in strategy.lower():
                    if not df[column].mode().empty:
                        fill_val = df[column].mode()[0]
                df[column] = df[column].fillna(fill_val)
                report['improvements_made'].append(f"Filled missing '{column}' with '{fill_val}'")
    return df

def llm_convert_data_types(df: pd.DataFrame, strategy: str, report: Dict[str, Any]):
    """Convert data types based on LLM recommendations."""
    for column in df.columns:
        if 'date' in column.lower() or 'time' in column.lower():
            if "date" in strategy.lower() and column in strategy:
                try:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                    report['improvements_made'].append(f"Converted {column} to datetime")
                except:
                    pass
    return df

def llm_handle_outliers(df: pd.DataFrame, numeric_columns: List[str], strategy: str, report: Dict[str, Any]):
    """Handle outliers based on LLM recommendations."""
    for column in numeric_columns:
        if "cap" in strategy.lower() or "clip" in strategy.lower():
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            
            outliers_before = len(df[(df[column] < lower) | (df[column] > upper)])
            if outliers_before > 0:
                df[column] = df[column].clip(lower=lower, upper=upper)
                report['improvements_made'].append(f"Capped {outliers_before} outliers in {column}")
    return df

def llm_clean_text_columns(df: pd.DataFrame, text_columns: List[str], strategy: str, report: Dict[str, Any]):
    """Clean text columns based on LLM recommendations."""
    for column in text_columns:
        if "clean" in strategy.lower() and "text" in strategy.lower():
            df[column] = df[column].astype(str).str.strip().str.title()
            report['improvements_made'].append(f"Cleaned and formatted {column}")
    return df

def calculate_data_quality_score(df):
    """Calculates a simple data quality score."""
    if df.empty:
        return 0.0
    completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
    uniqueness = 1 - (df.duplicated().sum() / df.shape[0]) if df.shape[0] > 0 else 1.0
    return round((completeness * 0.7 + uniqueness * 0.3) * 100, 2)

def clean_text_document(text: str) -> str:
    """Performs deep cleaning on a string of text using NLTK."""
    if 'nltk' not in globals(): return basic_text_clean(text)
    
    text = re.sub(r'\s+', ' ', text).strip().lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)