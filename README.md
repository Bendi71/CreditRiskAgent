# Credit Risk Agent 🏦🤖

An AI-powered credit risk assessment system that combines synthetic data generation, machine learning models, and multi-agent systems to automate credit analysis and generate comprehensive risk reports.

## 🎯 Project Objectives

This project demonstrates a complete end-to-end AI solution for credit risk assessment:

1. **Synthetic Data Generation**: Use OpenAI's GPT-4o mini to generate realistic credit application datasets
2. **Multi-Modal Model Training**: Train ML models on both structured financial data and unstructured text descriptions
3. **Automated Risk Assessment**: Deploy AI agents to analyze credit applications and generate professional reports
4. **Token-Based Probability Assessment**: Leverage OpenAI's token probabilities for nuanced risk evaluation

## 🏗️ Project Architecture

The project consists of three main components implemented as Jupyter notebooks:

```
CreditRiskAgent/
├── 1_generate_sample_data.ipynb     # AI-powered synthetic data generation
├── 2_train_llm_model.ipynb          # ML model training with embeddings
├── 3_ai_agent_credit_report.ipynb   # Multi-agent credit analysis system
├── requirements.txt                 # Python dependencies
├── credit_applications_dataset.json # Generated synthetic dataset
└── README.md                       # This file
```

## 📊 Data Generation Process

### Notebook 1: Generate Sample Data

**Key Features:**
- **Structured Data Generation**: Creates realistic credit applications with:
  - Applicant demographics (age, income, location, education)
  - Financial metrics (debt-to-income, employment length, credit history)
  - Loan details (amount, purpose, terms)
- **Unstructured Text Generation**: AI-generated loan application narratives
- **Token-Based Risk Assessment**: Uses OpenAI's token probabilities for default prediction
- **Economic Context Simulation**: Models good (20%) vs bad (80%) economic conditions

**Generated Features:**
```python
{
  "applicant_id": "APP_000001",
  "age": 35,
  "income": 75000,
  "loan_amount": 25000,
  "purpose": "debt_consolidation",
  "credit_history": "good",
  "employment_length": 8.5,
  "debt_to_income": 0.32,
  "location": "CA",
  "education": "bachelors",
  "text_description": "AI-generated loan application narrative...",
  "default_probability": 0.1234,
  "no_default_probability": 0.8766,
  "predicted_token": "ND"
}
```

## 🤖 Machine Learning Pipeline

### Notebook 2: Train LLM Model

**Modeling Approach:**
- **Feature Engineering**: Processes both structured financial data and text embeddings
- **Text Embeddings**: Converts loan application narratives to vector representations
- **Multi-Modal Training**: Combines structured features with text embeddings
- **Model Comparison**: Evaluates multiple algorithms (Logistic Regression, Random Forest, Neural Networks)
- **Performance Metrics**: Comprehensive evaluation with ROC curves, precision-recall, and feature importance

**Key Technologies:**
- Scikit-learn for traditional ML algorithms
- Transformers for text embeddings
- OpenAI embeddings for semantic understanding
- Feature scaling and preprocessing pipelines

**Key Goals:**
Traditional credit scoring relies only on structured data (income, credit score, etc.). This project proves that **the way applicants describe their loan needs contains valuable predictive information** that can significantly improve risk assessment accuracy.

## 🎯 AI Agent System

### Notebook 3: AI Agent Credit Report

**Multi-Agent Architecture:**
- **Data Analyst Agent**: Processes and validates credit application data
- **Credit Risk Analyst Agent**: Performs comprehensive risk assessment
- **Report Writer Agent**: Generates professional credit risk reports
- **Quality Assurance Agent**: Reviews and validates report accuracy

**Agent Capabilities:**
- Automated data validation and preprocessing
- Risk factor identification and scoring
- Professional report generation with recommendations
- Multi-agent collaboration and workflow orchestration

**Technologies Used:**
- CrewAI for multi-agent orchestration
- LangChain for agent tooling
- OpenAI GPT models for natural language processing
- Custom tools for credit analysis

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Jupyter Notebook environment
- Required Python packages (see requirements.txt)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd CreditRiskAgent
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up OpenAI API key:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Usage

1. **Generate Synthetic Data:**
```bash
jupyter notebook 1_generate_sample_data.ipynb
```
Run all cells to generate a synthetic credit application dataset.

2. **Train ML Models:**
```bash
jupyter notebook 2_train_llm_model.ipynb
```
Train and evaluate credit risk models on the generated data.

3. **Deploy AI Agents:**
```bash
jupyter notebook 3_ai_agent_credit_report.ipynb
```
Use the multi-agent system to analyze applications and generate reports.

## 📈 Key Features

### Advanced AI Techniques
- **Token Probability Analysis**: Leverages OpenAI's token-level probabilities for nuanced risk assessment
- **Economic Context Modeling**: Simulates different economic conditions in risk evaluation
- **Multi-Modal Learning**: Combines structured financial data with unstructured text analysis
- **Agent-Based Architecture**: Specialized AI agents for different aspects of credit analysis

### Comprehensive Risk Assessment
- Traditional credit risk factors (income, debt-to-income, credit history)
- Advanced text analysis of loan applications
- Economic context consideration
- Probabilistic risk scoring

### Professional Output
- Automated report generation
- Risk factor explanations
- Actionable recommendations
- Quality assurance validation

## 🔧 Technical Specifications

### Dependencies
- **Core ML**: scikit-learn, pandas, numpy
- **NLP**: transformers, openai, tiktoken
- **Visualization**: matplotlib, seaborn
- **Agents**: crewai, langchain
- **Data Processing**: json, datetime, tqdm

### Model Performance
The system generates and evaluates models on:
- 200+ synthetic credit applications
- Balanced representation of risk profiles
- Multiple evaluation metrics (AUC, precision, recall)
- Comparative analysis across algorithms

### Data Quality
- Realistic correlations between financial variables
- Diverse applicant demographics
- Varied loan purposes and amounts
- Economic condition modeling

## 📊 Project Outcomes

### Synthetic Data Generation
- Successfully generates realistic credit application data
- Maintains statistical relationships between variables
- Incorporates economic context into risk assessment
- Produces both structured and unstructured data

### Model Performance
- Achieves competitive performance on synthetic data
- Demonstrates value of text data in credit assessment
- Provides interpretable risk factors
- Enables probability-based decision making

### Agent System
- Automates complete credit analysis workflow
- Generates professional-quality reports
- Provides consistent, standardized assessments
- Scales to handle multiple applications

## 🤝 Contributing

This project serves as a demonstration of AI-powered credit risk assessment. Potential improvements include:

- Integration with real credit data (with proper privacy protections)
- Advanced embedding techniques for text analysis
- Ensemble methods for improved prediction accuracy
- Real-time risk monitoring capabilities
- Integration with external data sources

## ⚠️ Important Notes

- **Synthetic Data**: This project uses AI-generated synthetic data for demonstration purposes
- **Educational Use**: Intended for learning and research, not production credit decisions
- **API Costs**: OpenAI API usage will incur costs based on token consumption
- **Rate Limiting**: Includes appropriate delays to respect API rate limits

## 📄 License

This project is provided for educational and research purposes. Please ensure compliance with relevant financial regulations and data privacy requirements when adapting for real-world use.

## 🔗 Related Technologies

- **OpenAI GPT-4o mini**: For data generation and analysis
- **CrewAI**: Multi-agent system orchestration
- **LangChain**: Agent tooling and workflow management
- **Scikit-learn**: Traditional machine learning algorithms
- **Transformers**: Modern NLP embeddings

---

**Note**: This project demonstrates advanced AI techniques for credit risk assessment using synthetic data. Always ensure proper validation and compliance when working with real financial data.
