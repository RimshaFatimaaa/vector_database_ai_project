# 🕸️ LangGraph Implementation for AI Interview Coach

## Overview

This implementation adds advanced LangGraph workflow orchestration to the AI Interview Coach project. LangGraph provides powerful state management, workflow orchestration, and conversation persistence capabilities that enhance the interview simulation experience.

## 🚀 Features

### **Core Capabilities**
- **Workflow Orchestration**: Multi-step interview processes with state management
- **Memory Persistence**: Conversation history and session tracking
- **Error Handling**: Robust fallback mechanisms and graceful degradation
- **State Management**: Pydantic-based state models with validation
- **Integration**: Seamless integration with existing Streamlit app

### **Advanced Features**
- **Conditional Logic**: Dynamic workflow paths based on responses
- **Session Management**: Complete interview session tracking
- **Custom Metadata**: Flexible state extension capabilities
- **Fallback Mode**: Works without OpenAI API for testing/development

## 📁 File Structure

```
├── ai_modules/
│   └── langgraph_processor.py    # Core LangGraph implementation
├── notebooks/
│   └── langGraph_test.ipynb     # Enhanced notebook with examples
├── test_langgraph.py            # Comprehensive test suite
├── demo_langgraph.py            # Interactive demo script
└── app.py                       # Updated Streamlit app with LangGraph mode
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- OpenAI API key (optional, fallback mode available)
- All dependencies from `requirements.txt`

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

3. **Run Demo**
   ```bash
   python demo_langgraph.py
   ```

4. **Run Tests**
   ```bash
   python test_langgraph.py
   ```

5. **Start Streamlit App**
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage Examples

### Basic Usage

```python
from ai_modules.langgraph_processor import create_interview_processor

# Create processor
processor = create_interview_processor(use_openai=True)

# Run interview round
result = processor.run_interview_round(
    round_type="HR",
    context="Software engineer with 2 years experience",
    candidate_answer="I believe in teamwork and clear communication."
)

print(f"Question: {result['question']}")
print(f"Score: {result['score']}")
print(f"Evaluation: {result['evaluation']}")
```

### Advanced Usage with Custom State

```python
from ai_modules.langgraph_processor import InterviewState

# Create custom state
custom_state = InterviewState(
    round_type="Technical",
    context="Senior backend developer",
    metadata={
        "level": "senior",
        "specialization": "microservices",
        "experience_years": 5
    }
)

# Run workflow
result = processor.run_interview_round(
    round_type=custom_state.round_type,
    context=custom_state.context,
    candidate_answer="I would design a microservices architecture with API gateways..."
)
```

### Multi-Round Interview

```python
# Define interview scenarios
scenarios = [
    {"round": "HR", "context": "Senior engineer", "answer": "..."},
    {"round": "Technical", "context": "Backend role", "answer": "..."},
    {"round": "Behavioral", "context": "Team lead", "answer": "..."}
]

# Run multiple rounds
for scenario in scenarios:
    result = processor.run_interview_round(
        round_type=scenario["round"],
        context=scenario["context"],
        candidate_answer=scenario["answer"]
    )
    print(f"Round: {scenario['round']}, Score: {result['score']}")

# Get session summary
summary = processor.get_session_summary()
print(f"Total interactions: {summary['total_interactions']}")
print(f"Average score: {summary['average_score']:.1f}")
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No* | OpenAI API key for full functionality |
| `LANGCHAIN_API_KEY` | No | LangChain API key (if using LangSmith) |

*Required for full functionality, but fallback mode is available

### Model Configuration

```python
# Custom model configuration
processor = create_interview_processor(
    use_openai=True,
    model="gpt-4o-mini"  # or "gpt-4", "gpt-3.5-turbo"
)
```

### State Configuration

```python
# Custom state with metadata
state = InterviewState(
    round_type="Technical",
    context="Machine learning engineer",
    metadata={
        "session_id": "unique_id",
        "candidate_level": "senior",
        "specialization": "ML/DL"
    }
)
```

## 🧪 Testing

### Run All Tests
```bash
python test_langgraph.py
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Error Handling Tests**: Fallback mechanism validation
- **Mock Tests**: OpenAI integration testing

### Test Coverage
- ✅ State model validation
- ✅ Workflow orchestration
- ✅ Error handling and fallbacks
- ✅ Memory management
- ✅ Session tracking
- ✅ Integration with existing app

## 🎮 Demo & Examples

### Interactive Demo
```bash
python demo_langgraph.py
```

The demo includes:
- **Basic Workflow**: Simple question generation and evaluation
- **Multi-Round Interviews**: Multiple interview scenarios
- **State Management**: Custom state and metadata handling
- **Error Handling**: Fallback mode demonstration
- **Interactive Mode**: User-driven interview simulation

### Jupyter Notebook
Open `notebooks/langGraph_test.ipynb` for:
- **Step-by-step Examples**: Detailed workflow demonstrations
- **Advanced Features**: State management and custom workflows
- **Interactive Simulation**: Hands-on interview practice
- **Error Handling**: Fallback mode examples

## 🕸️ LangGraph Workflow

### Workflow Steps

1. **Question Generation**
   - Generate interview question based on round type and context
   - Use OpenAI GPT-4o-mini for high-quality questions
   - Fallback to template-based questions if OpenAI unavailable

2. **Response Evaluation**
   - Evaluate candidate response against question
   - Provide detailed feedback and scoring
   - Extract numerical scores from evaluation text

3. **History Management**
   - Add interaction to conversation history
   - Track session statistics
   - Maintain state persistence

### State Model

```python
class InterviewState(BaseModel):
    round_type: Optional[str] = None
    context: Optional[str] = None
    question: Optional[str] = None
    candidate_answer: Optional[str] = None
    evaluation: Optional[str] = None
    score: Optional[Dict[str, float]] = None
    feedback: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    current_step: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

## 🔌 Integration

### Streamlit App Integration

The LangGraph processor is integrated into the main Streamlit app with a new "LangGraph Workflow" mode:

1. **Mode Selection**: Choose between "Interview Simulation" and "LangGraph Workflow"
2. **Round Type**: Select HR, Technical, or Behavioral rounds
3. **Context Input**: Provide candidate background information
4. **Question Generation**: Generate questions using LangGraph workflow
5. **Response Evaluation**: Evaluate responses with detailed feedback
6. **Session Summary**: View comprehensive session statistics

### API Integration

```python
# Direct API usage
from ai_modules.langgraph_processor import LangGraphInterviewProcessor

processor = LangGraphInterviewProcessor(use_openai=True)
result = processor.run_interview_round(
    round_type="HR",
    context="Software engineer",
    candidate_answer="I believe in teamwork..."
)
```

## 📊 Performance Metrics

### Session Tracking
- **Total Interactions**: Number of Q&A pairs
- **Round Types**: Distribution of interview types
- **Average Score**: Overall performance metric
- **Memory Status**: Persistence availability

### Workflow Metrics
- **Success Rate**: Percentage of successful completions
- **Processing Time**: Workflow execution time
- **Error Frequency**: Fallback usage statistics
- **Model Usage**: OpenAI vs fallback mode usage

## 🛡️ Error Handling

### Fallback Mechanisms
- **No OpenAI API**: Falls back to template-based questions
- **API Errors**: Graceful degradation with error messages
- **Invalid Inputs**: Validation and error recovery
- **Network Issues**: Retry logic and timeout handling

### Error Types Handled
- ✅ Missing API keys
- ✅ Invalid state data
- ✅ Network timeouts
- ✅ Model errors
- ✅ Validation errors

## 🚀 Future Enhancements

### Planned Features
- **Custom Workflows**: User-defined workflow templates
- **Advanced Memory**: Long-term conversation persistence
- **Multi-Model Support**: Integration with other LLMs
- **Analytics Dashboard**: Advanced performance metrics
- **Workflow Templates**: Pre-built interview scenarios

### Extension Points
- **Custom Node Functions**: Add specialized processing steps
- **State Extensions**: Additional metadata fields
- **Workflow Modifications**: Custom workflow paths
- **Integration Hooks**: External system integration

## 📚 Documentation

### Additional Resources
- **API Documentation**: Detailed function references
- **Workflow Diagrams**: Visual workflow representations
- **Best Practices**: Usage guidelines and recommendations
- **Troubleshooting**: Common issues and solutions

### Support
- **Issues**: Report bugs and feature requests
- **Discussions**: Community support and questions
- **Examples**: Additional usage examples and tutorials

## 🎉 Conclusion

The LangGraph implementation provides a powerful, flexible, and robust foundation for advanced interview simulation workflows. With comprehensive error handling, state management, and integration capabilities, it enhances the AI Interview Coach with enterprise-grade workflow orchestration.

**Key Benefits:**
- ✅ **Advanced Workflow Management**: Multi-step processes with state persistence
- ✅ **Robust Error Handling**: Graceful degradation and fallback mechanisms
- ✅ **Seamless Integration**: Works with existing Streamlit app
- ✅ **Comprehensive Testing**: Full test coverage and validation
- ✅ **Interactive Demos**: Hands-on learning and experimentation
- ✅ **Extensible Design**: Easy to customize and extend

The implementation is production-ready and provides a solid foundation for building sophisticated interview simulation systems! 🚀
