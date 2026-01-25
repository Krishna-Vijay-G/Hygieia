<div align="center">
  <img src="public/logo.svg" alt="Hygieia Logo" width="200"/>
</div>

<div align="center">
  <h1>🤖 LLM Integration Documentation</h1>
  <strong>Dr. Hygieia AI Health Assistant - Intelligent Context-Aware Medical Guidance</strong><br/>
  <em>Hygieia AI Healthcare Platform</em>
</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Dr. Hygieia AI Assistant](#-dr-hygieia-ai-assistant)
3. [Analysis Summary Generation](#-analysis-summary-generation)
4. [Interactive Chat Sessions](#-interactive-chat-sessions)
5. [Context Awareness System](#-context-awareness-system)
6. [Architecture & Data Flow](#-architecture--data-flow)
7. [Safety & Guidelines](#-safety--guidelines)
8. [Technical Specifications](#-technical-specifications)

---

## 🎯 Overview

The Hygieia platform integrates advanced **Large Language Model (LLM)** capabilities to enhance the user experience through intelligent, context-aware interactions. Our AI assistant, **Dr. Hygieia**, provides personalized guidance, explains complex medical analysis results in simple terms, and offers empathetic support throughout the user's health journey.

### Key LLM Features

| Feature | Description |
|---------|-------------|
| **Analysis Summaries** | Automatic generation of human-readable summaries for all health analysis results |
| **Interactive Chat** | Real-time conversational interface for health-related questions |
| **Context Awareness** | Deep integration with user's analysis history and personal context |
| **Streaming Responses** | Real-time token-by-token response generation for better UX |
| **Fallback System** | Intelligent fallback responses when AI service is unavailable |

---

## 🩺 Dr. Hygieia AI Assistant

<div align="center">
  <img src="public/Chat.png" alt="Dr. Hygieia Chat Interface" width="800"/>
  <br/>
  <em>Dr. Hygieia Chat Interface - Context-Aware Medical Guidance</em>
</div>

### Assistant Personality

Dr. Hygieia is designed to be:

- **🤝 Empathetic** - Supportive and understanding in all interactions
- **📚 Educational** - Explains medical terms in simple, understandable language
- **⚠️ Responsible** - Always recommends consulting healthcare professionals
- **🔒 Privacy-Conscious** - Maintains patient confidentiality

### Capabilities

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dr. Hygieia Capabilities                     │
├─────────────────────────────────────────────────────────────────┤
│  ✓ Explain medical analysis results                             │
│  ✓ Provide educational health information                       │
│  ✓ Offer general wellness advice                                │
│  ✓ Answer questions about available analysis types              │
│  ✓ Guide users on when to seek professional help                │
│  ✓ Clarify medical terminology                                  │
│  ✓ Discuss risk factors and prevention strategies               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 Analysis Summary Generation

### Automatic Summary Creation

When a user completes any health analysis, the LLM automatically generates a personalized summary that:

1. **Explains the findings** in plain language
2. **Contextualizes the results** for the specific user
3. **Provides actionable recommendations** (always including professional consultation)

### Summary Structure

| Component | Description |
|-----------|-------------|
| **What was found** | Clear explanation of the analysis results |
| **What it means** | Interpretation in the context of user's health |
| **Next steps** | General recommendations and guidance |

### Example Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  User Submits   │────▶│  ML Model       │────▶│  LLM Generates  │
│  Analysis Data  │     │  Prediction     │     │  Summary        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  User Receives  │
                                               │  Results + AI   │
                                               │  Summary        │
                                               └─────────────────┘
```

### Supported Analysis Types

- **Heart Disease Risk Prediction** - Cardiovascular health assessment summaries
- **Diabetes Risk Prediction** - Diabetes risk factor explanations
- **Skin Condition Diagnosis** - Skin lesion analysis interpretations
- **Breast Cancer Risk Prediction** - Clinical risk assessment summaries
- **Breast Tissue Diagnosis** - FNA biopsy result explanations

---

## 💬 Interactive Chat Sessions

### Session Management

Users can engage in persistent chat sessions with Dr. Hygieia:

| Feature | Description |
|---------|-------------|
| **Session Persistence** | Conversations are saved and can be continued later |
| **Title Generation** | AI automatically generates descriptive titles for conversations |
| **Message History** | Full conversation history maintained for context |
| **Multi-Session** | Users can have multiple concurrent conversations |

### Chat Interface Features

```
┌────────────────────────────────────────────────────────────────┐
│                      Chat Session                               │
├────────────────────────────────────────────────────────────────┤
│ ┌──────────────┐                                               │
│ │  Sessions    │    ┌────────────────────────────────────┐     │
│ │  Sidebar     │    │        Message Area                │     │
│ │              │    │                                    │     │
│ │ • Session 1  │    │  🤖 Dr. Hygieia:                   │     │
│ │ • Session 2  │    │     Hello! I've reviewed your      │     │
│ │ • Session 3  │    │     recent analysis...             │     │
│ │              │    │                                    │     │
│ │ [+ New Chat] │    │  👤 User:                          │     │
│ │              │    │     What does this mean for me?    │     │
│ └──────────────┘    │                                    │     │
│                     │  🤖 Dr. Hygieia:                   │     │
│                     │     Let me explain...              │     │
│                     └────────────────────────────────────┘     │
│                     ┌────────────────────────────────────┐     │
│                     │  Type your message...        [Send]│     │
│                     └────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────┘
```

### Streaming Responses

The chat system supports **real-time streaming** of AI responses:

- Tokens appear as they are generated
- Provides immediate feedback to users
- Enhances the conversational experience
- Reduces perceived latency

---

## 🧠 Context Awareness System

### Multi-Level Context Integration

The LLM maintains awareness of multiple context levels:

```
                    ┌─────────────────────────────┐
                    │     SYSTEM CONTEXT          │
                    │  • Dr. Hygieia personality  │
                    │  • Medical guidelines       │
                    │  • Safety protocols         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     USER CONTEXT            │
                    │  • User's name              │
                    │  • Personalized greetings   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    ANALYSIS CONTEXT         │
                    │  • Analysis type            │
                    │  • Risk level               │
                    │  • Confidence score         │
                    │  • Input parameters         │
                    │  • Detailed results         │
                    │  • Analysis date            │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   CONVERSATION CONTEXT      │
                    │  • Message history          │
                    │  • Previous questions       │
                    │  • Follow-up awareness      │
                    └─────────────────────────────┘
```

### Analysis Context Data

When a chat session is linked to a specific analysis, the LLM receives:

| Data Point | Description |
|------------|-------------|
| **Analysis Type** | The type of health analysis performed |
| **Analysis Date** | When the analysis was conducted |
| **Risk Level** | The assessed risk level (Low/Medium/High) |
| **Confidence Score** | Model confidence in the prediction |
| **Risk Score** | Quantified risk percentage |
| **Input Parameters** | All user-provided input data |
| **Detailed Results** | Complete prediction output |
| **AI Summary** | Previously generated summary |

### Context-Aware Responses

The system automatically enriches every interaction:

```python
# Example Context Building
Context Components:
├── System Prompt (Dr. Hygieia guidelines)
├── User Profile (Name, preferences)
├── Analysis Details
│   ├── Type: "Heart Disease Risk"
│   ├── Date: "January 10, 2026"
│   ├── Risk Level: "Moderate"
│   ├── Confidence: "95.2%"
│   └── Input Parameters: {...}
└── Conversation History (last 20 messages)
```

---

## 🏗️ Architecture & Data Flow

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ Chat Page   │    │ Analysis    │    │ Formatted Message   │ │
│  │ Component   │    │ Result Page │    │ Component           │ │
│  └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘ │
└─────────┼──────────────────┼────────────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        BACKEND (Flask)                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Chat Routes API                           ││
│  │  /chat/sessions  /chat/message  /chat/stream               ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────────┐│
│  │              Dr. Hygieia Chat Service                       ││
│  │  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐ ││
│  │  │ Context        │  │ Summary        │  │ Chat          │ ││
│  │  │ Builder        │  │ Generator      │  │ Handler       │ ││
│  │  └────────────────┘  └────────────────┘  └───────────────┘ ││
│  └──────────────────────────┬──────────────────────────────────┘│
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │    LLM Provider     │
                    │  (Language Model)   │
                    └─────────────────────┘
```

### Data Flow Diagram

```
User Query → Context Assembly → LLM Processing → Response Generation → User Display
     │              │                 │                  │
     │              ├── User Info     │                  │
     │              ├── Analysis      │                  ├── Streamed
     │              ├── History       │                  │   Tokens
     │              └── System        │                  │
     │                  Prompt        │                  └── Complete
     │                                │                      Message
     └────────────────────────────────┴──────────────────────────────
```

---

## 🛡️ Safety & Guidelines

### AI Safety Protocols

The LLM integration follows strict safety guidelines:

| Guideline | Implementation |
|-----------|----------------|
| **No Diagnosis** | Never provides specific medical diagnoses |
| **No Treatment** | Never recommends specific treatments |
| **Professional Referral** | Always recommends consulting healthcare providers |
| **Clear Limitations** | Explicitly states AI limitations |
| **Privacy Protection** | Maintains confidentiality of user data |

### System Prompt Guidelines

```
┌─────────────────────────────────────────────────────────────────┐
│                    Safety Guidelines                            │
├─────────────────────────────────────────────────────────────────┤
│  ✓ Be empathetic and supportive                                 │
│  ✓ Explain medical terms simply                                 │
│  ✓ Recommend professional consultation                          │
│  ✓ Acknowledge AI limitations                                   │
│  ✗ Never provide specific diagnoses                             │
│  ✗ Never recommend specific treatments                          │
│  ✗ Never replace professional medical advice                    │
└─────────────────────────────────────────────────────────────────┘
```

### Fallback System

When the LLM service is unavailable, the system provides:

- Keyword-based intelligent responses
- Basic analysis result explanations
- Guidance to seek professional help
- Graceful degradation of service

---

## ⚙️ Technical Specifications

### Configuration

| Parameter | Value |
|-----------|-------|
| **Service Name** | Dr. Hygieia Chat Service |
| **Max Tokens (Summary)** | 500 |
| **Max Tokens (Chat)** | 1,000 |
| **Temperature** | 0.7 |
| **Context Window** | Last 20 messages |
| **Streaming** | Supported |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/status` | GET | Check AI service availability |
| `/chat/sessions` | GET | List user's chat sessions |
| `/chat/sessions` | POST | Create new chat session |
| `/chat/sessions/:id` | GET | Get session with messages |
| `/chat/sessions/:id/message` | POST | Send message |
| `/chat/sessions/:id/stream` | POST | Stream response |

### Response Format

```json
{
  "session": {
    "id": "uuid",
    "title": "Heart Health Discussion",
    "context_type": "analysis",
    "messages": [
      {
        "role": "assistant",
        "content": "Hello! I've reviewed your analysis..."
      }
    ]
  }
}
```

---

## 📊 Integration Benefits

| Benefit | Impact |
|---------|--------|
| **Improved Understanding** | Users better comprehend their health results |
| **Reduced Anxiety** | Empathetic AI reduces health-related stress |
| **Accessibility** | Complex medical data made accessible to all |
| **Engagement** | Interactive chat increases platform engagement |
| **Support** | 24/7 availability for health-related questions |

---

<div align="center">
  <br/>
  <strong>Hygieia AI Healthcare Platform</strong><br/>
  <em>Empowering Health Through Intelligent Technology</em><br/>
  <br/>
</div>
