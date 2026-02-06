"""
Dr. Hygieia AI Chat Service
===========================
Provides AI-powered chat functionality with context awareness for medical analysis results.
Uses Google Gemini API for intelligent responses about health analysis.
"""

import os
import json
from typing import List, Dict, Optional, Generator
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Check if Google Gemini is available
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None

# System prompt for Dr. Hygieia
SYSTEM_PROMPT = """You are Dr. Hygieia, an AI health assistant for the Hygieia Medical Diagnostic Platform. 
You provide helpful, accurate, and empathetic guidance about health-related topics and analysis results.

Your role is to:
1. Help users understand their medical analysis results from the Hygieia platform
2. Provide educational information about health conditions
3. Offer general wellness advice and healthy lifestyle recommendations
4. Answer questions about the analysis types available (heart disease risk, diabetes risk, skin conditions, breast cancer screening)
5. Guide users on when to seek professional medical advice

Important guidelines:
- Always be empathetic and supportive
- Never provide specific medical diagnoses or treatment recommendations
- Always recommend consulting with healthcare professionals for medical decisions
- Explain medical terms in simple, understandable language
- Be clear about the limitations of AI-based health analysis
- Maintain patient privacy and confidentiality
- If discussing analysis results, explain what the metrics mean in context

Analysis types you can help with:
- Heart Disease Risk Prediction: Uses clinical parameters to assess cardiovascular risk
- Diabetes Risk Prediction: Evaluates risk factors for diabetes
- Skin Condition Diagnosis: Analyzes skin lesion images for potential conditions
- Breast Cancer Screening: Both predictive risk assessment and diagnostic analysis

Formatting guidelines:
- Use numbered lists (1. 2. 3.) for sequential steps or ordered information
- Use dash bullets (- item) for unordered lists, NOT asterisks
- Use **bold** for emphasis on important terms
- Keep paragraphs concise and well-spaced
- Use natural, conversational language

Remember: You are an educational tool, not a replacement for professional medical advice."""


def get_analysis_context(analysis: dict) -> str:
    """Generate context string from analysis data"""
    if not analysis:
        return ""
    
    analysis_type = analysis.get('analysis_type', 'unknown')
    result = analysis.get('result', {})
    risk_level = analysis.get('risk_level', 'unknown')
    confidence = analysis.get('confidence')
    risk_score = analysis.get('risk_score')
    input_data = analysis.get('input_data', {})
    created_at = analysis.get('created_at')
    
    context_parts = [
        f"\n--- Analysis Context ---",
        f"Analysis Type: {analysis_type}",
        f"Risk Level: {risk_level}",
    ]
    
    if created_at:
        context_parts.append(f"Analysis Date: {created_at}")
    
    if confidence is not None:
        # Confidence is expected as a fraction (0-1). If a raw percent (e.g. 75) is provided,
        # detect and format accordingly.
        if isinstance(confidence, (int, float)) and confidence > 1:
            context_parts.append(f"Confidence: {confidence:.1f}%")
        else:
            context_parts.append(f"Confidence: {confidence:.1%}")

    if risk_score is not None:
        # Some models return a risk_score as a fraction (0-1) while others return a percentage (0-100).
        # Format defensively: if the value is > 1 assume it's already a percent and display directly,
        # otherwise format as a percentage.
        if isinstance(risk_score, (int, float)) and risk_score > 1:
            context_parts.append(f"Risk Score: {risk_score:.1f}%")
        else:
            context_parts.append(f"Risk Score: {risk_score:.1%}")
    
    # Include input parameters - this is critical for the AI to understand what the user provided
    if input_data and isinstance(input_data, dict):
        context_parts.append("\n--- User's Input Parameters ---")
        for key, value in input_data.items():
            # Format the key nicely
            display_key = key.replace('_', ' ').replace('actual ', '').title()
            context_parts.append(f"{display_key}: {value}")
        context_parts.append("--- End Input Parameters ---")
    
    # Add specific result details based on analysis type
    if analysis_type == 'heart-prediction':
        context_parts.append("\nThis is a heart disease risk prediction analysis.")
        if isinstance(result, dict):
            prediction = result.get('prediction', result.get('risk_level', 'unknown'))
            context_parts.append(f"Prediction: {prediction}")
            
    elif analysis_type == 'diabetes-prediction':
        context_parts.append("\nThis is a diabetes risk prediction analysis.")
        if isinstance(result, dict):
            prediction = result.get('prediction', result.get('risk_level', 'unknown'))
            context_parts.append(f"Prediction: {prediction}")
            
    elif analysis_type == 'skin-diagnosis':
        context_parts.append("\nThis is a skin condition diagnosis analysis.")
        if isinstance(result, dict):
            condition = result.get('predicted_condition', result.get('condition', result.get('condition_name', 'unknown')))
            context_parts.append(f"Detected Condition: {condition}")
            
    elif analysis_type in ['breast-prediction', 'breast-diagnosis']:
        context_parts.append(f"\nThis is a breast cancer {'risk prediction' if 'prediction' in analysis_type else 'diagnostic'} analysis.")
        if isinstance(result, dict):
            prediction = result.get('prediction', result.get('diagnosis', result.get('condition_name', 'unknown')))
            context_parts.append(f"Result: {prediction}")
    
    context_parts.append("--- End Context ---\n")
    return "\n".join(context_parts)


def get_user_context(user: dict) -> str:
    """Generate context string from user data"""
    if not user:
        return ""
    
    return f"\nUser: {user.get('first_name', 'User')} {user.get('last_name', '')}\n"


class DrHygieiaChat:
    """Dr. Hygieia AI Chat Service"""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
        self.client = None
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print(f"Google Gemini client initialized with model: {self.model_name}")
            except Exception as e:
                print(f"Error initializing Gemini client: {e}")
                self.client = None

    @property
    def is_available(self) -> bool:
        """Check if Gemini AI service is available"""
        return self.client is not None
    
    def generate_analysis_summary(self, analysis: dict, user: dict = None) -> str:
        """Generate an AI summary for an analysis result"""
        if not self.is_available:
            return self._get_fallback_summary(analysis)
        
        analysis_context = get_analysis_context(analysis)
        user_context = get_user_context(user) if user else ""
        
        prompt = f"""Based on the following medical analysis result, provide a brief, friendly summary 
that explains the results in simple terms. Include:
1. What the analysis found
2. What this means for the user
3. Any general recommendations (always recommend consulting a healthcare professional)

{user_context}
{analysis_context}

Please provide a concise summary (2-3 paragraphs) that is informative yet reassuring."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=512,
                    temperature=0.7
                )
            )
            
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                return self._get_fallback_summary(analysis)
        except Exception as e:
            print(f"Error in Gemini summary: {e}")
            return self._get_fallback_summary(analysis)
    
    def _get_fallback_summary(self, analysis: dict) -> str:
        """Generate a fallback summary when AI is not available"""
        analysis_type = analysis.get('analysis_type', 'unknown')
        risk_level = analysis.get('risk_level', 'unknown')
        confidence = analysis.get('confidence')
        
        type_names = {
            'heart-prediction': 'Heart Disease Risk',
            'diabetes-prediction': 'Diabetes Risk',
            'skin-diagnosis': 'Skin Condition',
            'breast-prediction': 'Breast Cancer Risk',
            'breast-diagnosis': 'Breast Cancer Diagnostic'
        }
        
        type_name = type_names.get(analysis_type, analysis_type)
        
        summary = f"Your {type_name} analysis has been completed. "
        
        if risk_level:
            summary += f"The assessment indicates a {risk_level.lower()} risk level. "
        
        if confidence:
            summary += f"This result was determined with {confidence:.1%} confidence. "
        
        summary += "\n\nPlease note that this analysis is for informational purposes only and should not replace professional medical advice. "
        summary += "We recommend discussing these results with your healthcare provider for personalized guidance."
        
        return summary
    
    def chat(
        self, 
        messages: List[Dict], 
        analysis: dict = None, 
        user: dict = None,
        stream: bool = False
    ) -> str | Generator:
        """
        Send a chat message and get a response
        """
        if not self.is_available:
            return self._get_fallback_response(messages)
        
        # Build context
        system_content = SYSTEM_PROMPT
        if user:
            system_content += get_user_context(user)
        if analysis:
            system_content += get_analysis_context(analysis)
        
        # Convert messages to Gemini Content format
        # Skip system messages as we use system_instruction instead
        history = []
        for msg in messages[-20:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role in ['user', 'model'] and content:
                history.append(types.Content(
                    role=role,
                    parts=[types.Part(text=content)]
                ))
        
        if stream:
            return self._stream_response(history, system_content)
        
        try:
            # Use the chat session for multi-turn conversation
            chat = self.client.chats.create(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=system_content,
                    max_output_tokens=1000,
                    temperature=0.7
                ),
                history=history[:-1] if len(history) > 1 else []
            )
            
            # Send the last message
            last_message = history[-1].parts[0].text if history else "Hello"
            response = chat.send_message(last_message)
            
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                return self._get_fallback_response(messages)
        except Exception as e:
            print(f"Error in Gemini chat: {e}")
            return self._get_fallback_response(messages)
    
    def _stream_response(self, history: List, system_content: str) -> Generator:
        """Stream the response token by token"""
        try:
            # Use the chat session for streaming
            chat = self.client.chats.create(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=system_content,
                    max_output_tokens=1000,
                    temperature=0.7
                ),
                history=history[:-1] if len(history) > 1 else []
            )
            
            # Send the last message with streaming
            last_message = history[-1].parts[0].text if history else "Hello"
            stream = chat.send_message_stream(last_message)
            
            for chunk in stream:
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    text = chunk.candidates[0].content.parts[0].text
                    if text:
                        yield text
        except Exception as e:
            print(f"Gemini Streaming error: {e}")
            yield "I apologize, but I'm having trouble generating a response right now."

    def _get_fallback_response(self, messages: List[Dict]) -> str:
        """Generate a fallback response when AI is not available"""
        last_message = messages[-1].get('content', '') if messages else ''
        
        # Basic keyword-based responses
        lower_msg = last_message.lower()
        
        if any(word in lower_msg for word in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! I'm Dr. Hygieia, your AI health assistant. While my full AI capabilities are currently limited, I'm here to help you understand your health analysis results. How can I assist you today?"
        
        if any(word in lower_msg for word in ['heart', 'cardiovascular', 'cardiac']):
            return "Heart health is crucial for overall wellbeing. Our heart disease risk analysis evaluates various clinical parameters to assess your cardiovascular risk. For specific medical advice about your heart health, please consult with a cardiologist or your primary care physician."
        
        if any(word in lower_msg for word in ['diabetes', 'blood sugar', 'glucose']):
            return "Diabetes risk assessment helps identify potential risk factors for developing diabetes. Managing blood sugar through diet, exercise, and regular monitoring is important. Please consult with your healthcare provider for personalized diabetes management advice."
        
        if any(word in lower_msg for word in ['skin', 'lesion', 'mole', 'rash']):
            return "Our skin diagnosis analysis uses AI to evaluate skin lesions and identify potential conditions. However, any concerning skin changes should always be examined by a dermatologist for proper diagnosis and treatment."
        
        if any(word in lower_msg for word in ['breast', 'cancer', 'mammogram']):
            return "Breast cancer screening is an important part of preventive healthcare. Our analysis tools can help assess risk and aid in detection, but regular mammograms and clinical breast exams with healthcare professionals are essential for comprehensive screening."
        
        if any(word in lower_msg for word in ['result', 'analysis', 'report']):
            return "I'd be happy to help you understand your analysis results. Your results page shows the risk level, confidence score, and detailed findings. Remember that these results are informational and should be discussed with your healthcare provider for proper interpretation and next steps."
        
        return "Thank you for your question. While my full AI capabilities are temporarily limited, I encourage you to explore your analysis results on the platform. For specific medical questions, please consult with a qualified healthcare professional. Is there anything specific about the Hygieia platform I can help you with?"
    
    def generate_title(self, messages: List[Dict]) -> str:
        """Generate a title for a chat session based on the conversation"""
        if not self.is_available or not messages:
            return "New Conversation"
        
        # Get the first user message
        first_user_msg = next(
            (msg.get('content', '') for msg in messages if msg.get('role') == 'user'),
            None
        )
        
        if not first_user_msg:
            return "New Conversation"

        title_prompt = f"Generate a brief, descriptive title (max 5 words) for a conversation that starts with: {first_user_msg[:500]}\n\nReturn only the title, no quotes or punctuation."

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=title_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=20,
                    temperature=0.5
                )
            )
            
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                title = response.candidates[0].content.parts[0].text.strip()
                return title[:50] if title else "New Conversation"
            else:
                return "New Conversation"
        except Exception as e:
            print(f"Error in Gemini title generation: {e}")
            return "New Conversation"


# Singleton instance
dr_hygieia = DrHygieiaChat()
