import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional
import os
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

class ColumnRequest(BaseModel):
    name: str
    reason: str

class ModelAnalysis(BaseModel):
    requested_columns: List[ColumnRequest]
    analysis_type: str
    formulas_needed: List[str]
    sample_size_recommendation: int

class CSVSchemaAnalyzer:
    def __init__(self, csv_path: str, sample_size: int = 1000):
        self.csv_path = csv_path
        self.sample_size = sample_size
        self.schema = None
        
    def generate_schema_digest(self) -> Dict[str, Any]:
        """
        Step 1: Generate lightweight schema digest from CSV sample
        """
        print(f"Analyzing schema from {self.csv_path}...")
        
        # Read a sample of the CSV to analyze structure
        df_sample = pd.read_csv(self.csv_path, nrows=self.sample_size)
        
        # Get total row count (approximate)
        total_rows = self._estimate_total_rows()
        
        columns_info = []
        for col in df_sample.columns:
            col_info = {
                "name": col,
                "dtype": self._get_dtype_string(df_sample[col]),
                "nulls": int(df_sample[col].isnull().sum())
            }
            
            # Add type-specific statistics
            if df_sample[col].dtype in ['int64', 'float64']:
                col_info.update({
                    "mean": round(float(df_sample[col].mean()), 2) if not df_sample[col].isnull().all() else None,
                    "min": float(df_sample[col].min()) if not df_sample[col].isnull().all() else None,
                    "max": float(df_sample[col].max()) if not df_sample[col].isnull().all() else None
                })
            elif df_sample[col].dtype == 'object':
                # Get top 3 most frequent values
                top_values = df_sample[col].value_counts().head(3).index.tolist()
                col_info["top_3"] = [str(val)[:20] + "..." if len(str(val)) > 20 else str(val) 
                                   for val in top_values]
            elif 'datetime' in str(df_sample[col].dtype):
                col_info.update({
                    "min_date": str(df_sample[col].min()) if not df_sample[col].isnull().all() else None,
                    "max_date": str(df_sample[col].max()) if not df_sample[col].isnull().all() else None
                })
            
            columns_info.append(col_info)
        
        # Create schema digest
        self.schema = {
            "table": {
                "id": os.path.basename(self.csv_path).replace('.csv', ''),
                "n_rows": total_rows,
                "n_cols": len(df_sample.columns),
                "file_size_mb": round(os.path.getsize(self.csv_path) / (1024*1024), 2),
                "columns": columns_info
            }
        }
        
        return self.schema
    
    def _estimate_total_rows(self) -> int:
        """Estimate total rows in CSV without loading entire file"""
        with open(self.csv_path, 'r') as f:
            # Count lines in first chunk
            chunk_size = 8192
            chunk = f.read(chunk_size)
            lines_in_chunk = chunk.count('\n')
            
            # Get file size and estimate
            file_size = os.path.getsize(self.csv_path)
            estimated_rows = int((file_size / chunk_size) * lines_in_chunk) - 1  # -1 for header
            
        return max(estimated_rows, self.sample_size)
    
    def _get_dtype_string(self, series: pd.Series) -> str:
        """Convert pandas dtype to readable string"""
        dtype_str = str(series.dtype)
        if 'int' in dtype_str:
            return 'integer'
        elif 'float' in dtype_str:
            return 'float'
        elif 'datetime' in dtype_str:
            return 'date'
        elif 'bool' in dtype_str:
            return 'boolean'
        else:
            return 'string'
    
    def print_schema_digest(self):
        """Pretty print the schema digest"""
        if not self.schema:
            print("No schema generated yet. Run generate_schema_digest() first.")
            return
            
        print("\n" + "="*50)
        print("SCHEMA DIGEST")
        print("="*50)
        print(json.dumps(self.schema, indent=2))
        print("="*50)

class OpenAIAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def analyze_schema_for_task(self, schema: Dict[str, Any], task_description: str) -> ModelAnalysis:
        print(f"🤖 Sending schema to OpenAI for analysis...")
        
        system_prompt = """You are an expert data analyst. Given a dataset schema and a task description, 
        you need to identify the minimum set of columns required to complete the task effectively.
        
        Consider:
        - Which columns are essential vs nice-to-have
        - Data types and their relevance to the task
        - Potential relationships between columns
        - Computational efficiency (fewer columns = faster processing)
        
        Be selective - only choose columns that are truly necessary."""
        
        user_prompt = f"""
        Dataset Schema:
        {json.dumps(schema, indent=2)}
        
        Task: {task_description}
        
        Analyze this schema and identify:
        1. The minimum set of columns needed for this task effectively
        2. The reason for selecting each column
        3. What type of analysis this enables
        4. Any formulas or calculations you would apply
        5. Recommended sample size for initial analysis
        """
        
        try:
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=ModelAnalysis
            )
            
            analysis = response.choices[0].message.parsed
            print(f"✅ Model analysis complete!")
            return analysis
            
        except Exception as e:
            print(f"❌ Error calling OpenAI API: {e}")
            # Fallback to simple column selection
            available_columns = [col['name'] for col in schema['table']['columns']]
            fallback_columns = [
                ColumnRequest(name=col, reason="Fallback selection") 
                for col in available_columns[:5]
            ]
            return ModelAnalysis(
                requested_columns=fallback_columns,
                analysis_type="Fallback analysis",
                formulas_needed=["Basic statistical analysis"],
                sample_size_recommendation=1000
            )

class CSVDataSlicer:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
    
    def extract_columns(self, requested_columns: List[str], 
                       max_rows: Optional[int] = None,
                       sample_strategy: str = 'head') -> pd.DataFrame:
        """
        Step 3: Extract only the requested columns from CSV

        Args:
            requested_columns: List of column names to extract
            max_rows: Maximum number of rows to return (None for all)
            sample_strategy: 'head', 'tail', 'random', or 'sample'
        """
        print(f"📊 Extracting columns: {requested_columns}")
        
        # Validate columns exist
        df_sample = pd.read_csv(self.csv_path, nrows=1)
        available_columns = df_sample.columns.tolist()
        valid_columns = [col for col in requested_columns if col in available_columns]
        
        if len(valid_columns) != len(requested_columns):
            missing = set(requested_columns) - set(valid_columns)
            print(f"⚠️  Warning: Columns not found: {missing}")
        
        if sample_strategy == 'head':
            df = pd.read_csv(self.csv_path, usecols=valid_columns, nrows=max_rows)
        elif sample_strategy == 'tail':
            # For tail, we need to read all and take last rows
            df_full = pd.read_csv(self.csv_path, usecols=valid_columns)
            df = df_full.tail(max_rows) if max_rows else df_full
        elif sample_strategy == 'random':
            # Read with skiprows for random sampling
            total_rows = sum(1 for line in open(self.csv_path)) - 1  # -1 for header
            if max_rows and max_rows < total_rows:
                skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                                  total_rows - max_rows, 
                                                  replace=False))
                df = pd.read_csv(self.csv_path, usecols=valid_columns, skiprows=skip_rows)
            else:
                df = pd.read_csv(self.csv_path, usecols=valid_columns)
        else:  # sample strategy
            df = pd.read_csv(self.csv_path, usecols=valid_columns, nrows=max_rows)
        
        print(f"✅ Extracted {len(df)} rows with {len(df.columns)} columns")
        return df

def intelligent_csv_analysis(csv_path: str, openai_api_key: str, task_description: str):
    """
    Complete intelligent workflow with OpenAI model integration
    
    Args:
        csv_path: Path to your CSV file
        openai_api_key: Your OpenAI API key
        task_description: What you want to do with the data
    """
    print("🚀 Starting Intelligent Schema-Then-Slice Workflow")
    print("-" * 60)
    
    # Step 1: Generate Schema Digest
    print("\n📋 STEP 1: Analyzing Dataset Schema")
    print("-" * 40)
    analyzer = CSVSchemaAnalyzer(csv_path, sample_size=1000)
    schema = analyzer.generate_schema_digest()
    analyzer.print_schema_digest()
    
    # Step 2: AI-Powered Column Selection
    print(f"\n🤖 STEP 2: AI Analysis for Task: '{task_description}'")
    print("-" * 50)
    ai_analyzer = OpenAIAnalyzer(openai_api_key)
    analysis = ai_analyzer.analyze_schema_for_task(schema, task_description)
    
    # Print AI analysis results
    print(f"\n🎯 AI Analysis Results:")
    print(f"Analysis Type: {analysis.analysis_type}")
    print(f"Recommended Sample Size: {analysis.sample_size_recommendation:,} rows")
    
    print(f"\n📊 Selected Columns ({len(analysis.requested_columns)}):")
    for col_req in analysis.requested_columns:
        print(f"  • {col_req.name}: {col_req.reason}")
    
    print(f"\n🧮 Suggested Formulas/Calculations:")
    for formula in analysis.formulas_needed:
        print(f"  • {formula}")
    
    # Step 3: Extract AI-Selected Data
    print(f"\n📊 STEP 3: Extracting AI-Selected Data")
    print("-" * 40)
    
    slicer = CSVDataSlicer(csv_path)
    requested_column_names = [col.name for col in analysis.requested_columns]
    
    # Use AI's recommended sample size
    sliced_data = slicer.extract_columns(
        requested_columns=requested_column_names,
        max_rows=analysis.sample_size_recommendation,
        sample_strategy='head'
    )
    
    print(f"\n✅ RESULTS:")
    print(f"Original file size: {os.path.getsize(csv_path) / 1024**2:.2f} MB")
    print(f"Final dataset shape: {sliced_data.shape}")
    print(f"Final dataset size: {sliced_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Data reduction: {(1 - sliced_data.memory_usage(deep=True).sum() / os.path.getsize(csv_path)) * 100:.1f}%")
    
    print(f"\n📋 Sample of extracted data:")
    print(sliced_data.head())
    
    return sliced_data, analysis

if __name__ == "__main__":
    CSV_FILE_PATH = "./data/files/zoomcar-opening-balance.csv" 
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
    TASK_DESCRIPTION = "build a 3-month revenue forecast model" 
    
    
    try:
        # Run the intelligent workflow
        final_data, ai_analysis = intelligent_csv_analysis(
            csv_path=CSV_FILE_PATH,
            openai_api_key=OPENAI_API_KEY,
            task_description=TASK_DESCRIPTION
        )
        
        print(f"\n🎯 SUCCESS: AI-optimized dataset ready!")
        print(f"You can now use this focused dataset for: {ai_analysis.analysis_type}")
        
    except FileNotFoundError:
        print(f"❌ File not found: {CSV_FILE_PATH}")
        print("Please update CSV_FILE_PATH with your actual CSV file path")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your OpenAI API key is valid and you have sufficient credits")
