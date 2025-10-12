# 🤖 AI Interview Coach

An intelligent interview coaching application powered by advanced NLP and LLM technology. This application helps candidates practice for job interviews by generating relevant questions and providing detailed feedback on their responses.

## ✨ Features

- **🎭 Interview Simulation**: Generate questions based on job type and difficulty level
- **🧠 AI-Powered Analysis**: Advanced NLP processing and LLM evaluation
- **📊 Detailed Feedback**: Comprehensive scoring and improvement suggestions
- **🔐 Secure Authentication**: User authentication system
- **📈 Session Tracking**: Monitor progress and performance over time
- **🎯 Multiple Question Types**: HR behavioral and technical questions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for AI features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/RimshaFatimaaa/AI-Interview-Coach.git
   cd AI-Interview-Coach
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=your-openai-api-key-here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Required for AI features
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Supabase Configuration (if using database features)
SUPABASE_URL=your-supabase-url-here
SUPABASE_KEY=your-supabase-key-here
```

### Getting an OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and add it to your `.env` file

## 📁 Project Structure

```
AI-Interview-Coach/
├── ai_modules/                 # Core AI processing modules
│   ├── auth.py                # Authentication system
│   ├── auth_ui.py             # Authentication UI components
│   ├── langchain_processor.py # LangChain integration
│   ├── llm_processor.py       # LLM processing
│   ├── llm_processor_simple.py # Simplified LLM processor
│   └── nlp_processor.py       # NLP processing
├── notebooks/                  # Jupyter notebooks for testing
│   ├── LLMs_test.ipynb        # LLM testing notebook
│   ├── LLMs_test.py           # LLM testing script
│   ├── langchain.ipynb        # LangChain integration notebook
│   └── nlp_test.ipynb         # NLP testing notebook
├── pages/                      # Static assets
│   └── Ai-coach.jpg           # Application logo
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🎯 Usage

### Basic Workflow

1. **Start the Application**: Run `streamlit run app.py`
2. **Authentication**: Log in with your credentials
3. **Select Mode**: Choose "Interview Simulation" mode
4. **Configure Settings**: Select question type and difficulty
5. **Generate Question**: Click "Generate Question" to get an interview question
6. **Provide Response**: Enter your answer in the text area
7. **Analyze Response**: Click "Analyze Response" to get detailed feedback

### Question Types

- **HR Behavioral**: Questions about past experiences, teamwork, leadership
- **Technical**: Questions about technical skills, problem-solving, coding

### Difficulty Levels

- **Easy**: Basic questions suitable for entry-level positions
- **Medium**: Intermediate questions for mid-level positions
- **Hard**: Advanced questions for senior-level positions

## 🔍 Features in Detail

### AI-Powered Analysis

The application uses multiple AI technologies:

- **NLP Processing**: Text preprocessing, feature extraction, sentiment analysis
- **LLM Evaluation**: OpenAI GPT models for intelligent response evaluation
- **LangChain Integration**: Advanced conversation management and memory

### Scoring System

Responses are evaluated on multiple criteria:

- **Relevance**: How well the answer addresses the question
- **Completeness**: Thoroughness of the response
- **Clarity**: How clear and well-structured the answer is
- **Overall Score**: Weighted average of all criteria

### Feedback Features

- **Detailed Metrics**: Individual scores for each evaluation criterion
- **Constructive Feedback**: Specific suggestions for improvement
- **Session Summary**: Track progress over multiple questions
- **Reference Answers**: Compare with canonical answers when available

## 🛠️ Development

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test files
python test_nlp.py
python test_llm.py
python test_auth.py
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make your changes
3. Test thoroughly
4. Commit changes: `git commit -m "Add new feature"`
5. Push to GitHub: `git push origin feature/new-feature`
6. Create pull request

## 📊 Performance

The application is optimized for:

- **Fast Response Times**: Efficient NLP processing and caching
- **Scalability**: Modular design for easy scaling
- **Memory Management**: Optimized for long interview sessions
- **Error Handling**: Robust error handling and fallback mechanisms

## 🔒 Security

- **API Key Protection**: Environment variables for sensitive data
- **Input Validation**: Comprehensive input sanitization
- **Authentication**: Secure user authentication system
- **Data Privacy**: No sensitive data stored permanently

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues:

1. Check the [Issues](https://github.com/RimshaFatimaaa/AI-Interview-Coach/issues) page
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

## 🙏 Acknowledgments

- OpenAI for providing the GPT models
- LangChain for conversation management
- Streamlit for the web interface
- The open-source community for various libraries and tools

## 📈 Roadmap

- [ ] Multi-language support
- [ ] Video interview simulation
- [ ] Advanced analytics dashboard
- [ ] Integration with job boards
- [ ] Mobile application
- [ ] Voice-to-text input

---

**Built with ❤️ using Streamlit, Transformers, and LangChain**
