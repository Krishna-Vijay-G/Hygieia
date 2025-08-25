"""
AI Chatbot service for medical consultations with image analysis capabilities.
"""
import os
import logging
from google import genai
from google.genai import types
from typing import Dict, List, Optional, Tuple
import base64
from PIL import Image
import io
import dotenv

dotenv.load_dotenv()

# Initialize Gemini client with error handling
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = None

if gemini_api_key and gemini_api_key != "your_gemini_api_key_here":
    try:
        client = genai.Client(api_key=gemini_api_key)
        print("Gemini API client initialized successfully for chatbot")
    except Exception as e:
        print(f"Error initializing Gemini client for chatbot: {e}")
        client = None
else:
    print("Gemini API key not set. Chatbot will use fallback responses only.")

class MedicalChatbot:
    """AI-powered medical chatbot with image analysis capabilities"""
    
    def __init__(self):
        self.system_prompt = """You are Dr. Hygieia, an AI medical assistant designed to provide educational health information and analyze medical diagnostic results. Your role is to:

1. EDUCATIONAL SUPPORT: Provide clear, patient-friendly explanations of medical conditions, symptoms, and general health guidance.

2. DIAGNOSTIC ANALYSIS: When presented with diagnostic results or medical images, analyze them and provide:
   - Clinical interpretation of findings
   - Risk assessment and significance
   - Recommended follow-up actions
   - When to seek immediate medical attention

3. PRESCRIPTION GUIDANCE: When appropriate, suggest general treatment approaches, lifestyle modifications, and preventive measures. Always emphasize that actual prescriptions must come from licensed healthcare providers.

4. SAFETY GUIDELINES:
   - Always include medical disclaimers
   - Emphasize the need for professional medical evaluation
   - Never provide specific drug dosages or replace clinical judgment
   - Encourage immediate medical attention for urgent symptoms

5. COMMUNICATION STYLE:
   - Use clear, accessible language
   - Be empathetic and supportive
   - Provide structured, actionable information
   - Include relevant medical context when helpful

Remember: You are an educational tool to complement, not replace, professional medical care."""

    def generate_response(self, user_message: str, conversation_history: List[Dict], 
                         diagnostic_context: Optional[Dict] = None, 
                         image_data: Optional[bytes] = None) -> str:
        """Generate AI response with context"""
        # Use fallback response if Gemini client is not available
        if client is None:
            return self._get_fallback_response()
            
        try:
            # Build conversation context
            messages = []
            
            # Add diagnostic context if available
            if diagnostic_context:
                context_message = self._build_diagnostic_context(diagnostic_context)
                messages.append(types.Content(role="user", parts=[types.Part(text=context_message)]))
            
            # Add conversation history (last 10 messages for context)
            for msg in conversation_history[-10:]:
                role = "user" if msg['role'] == 'user' else "model"
                messages.append(types.Content(role=role, parts=[types.Part(text=msg['content'])]))
            
            # Prepare current message
            current_parts = [types.Part(text=user_message)]
            
            # Add image if provided
            if image_data:
                current_parts.append(types.Part.from_bytes(
                    data=image_data,
                    mime_type="image/jpeg"
                ))
            
            messages.append(types.Content(role="user", parts=current_parts))
            
            # Generate response
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.3,
                    max_output_tokens=1500
                )
            )
            
            if response.text:
                return self._format_medical_response(response.text)
            else:
                return self._get_fallback_response()
                
        except Exception as e:
            logging.error(f"Error generating chatbot response: {str(e)}")
            return self._get_fallback_response()
    
    def analyze_diagnostic_result(self, diagnostic_result: Dict, image_path: Optional[str] = None) -> str:
        """Provide detailed analysis of diagnostic results"""
        # Use fallback analysis if Gemini client is not available
        if client is None:
            return self._get_fallback_analysis()
            
        try:
            analysis_prompt = f"""
            Please provide a comprehensive medical analysis of the following diagnostic result:
            
            **Diagnostic Type:** {diagnostic_result.get('result_type', 'Unknown')}
            **Prediction:** {diagnostic_result.get('prediction', {})}
            **Input Parameters:** {diagnostic_result.get('input_data', {})}
            
            Please provide:
            1. **Clinical Interpretation:** What these results mean in medical terms
            2. **Risk Assessment:** Severity and implications of these findings
            3. **Recommended Actions:** Immediate and long-term steps to take
            4. **Treatment Considerations:** General approaches to management
            5. **Follow-up Care:** What monitoring or additional tests might be needed
            6. **Lifestyle Recommendations:** Preventive measures and health optimization
            
            Include appropriate medical disclaimers and emphasize the need for professional medical validation.
            """
            
            # Prepare message parts
            parts = [types.Part(text=analysis_prompt)]
            
            # Add image if available
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as img_file:
                    image_data = img_file.read()
                    parts.append(types.Part.from_bytes(
                        data=image_data,
                        mime_type="image/jpeg"
                    ))
                    parts.append(types.Part(text="Please also analyze this medical image in detail."))
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.2,
                    max_output_tokens=2000
                )
            )
            
            if response.text:
                return self._format_medical_response(response.text)
            else:
                return self._get_fallback_analysis()
                
        except Exception as e:
            logging.error(f"Error analyzing diagnostic result: {str(e)}")
            return self._get_fallback_analysis()
    
    def _build_diagnostic_context(self, diagnostic_context: Dict) -> str:
        """Build context message from diagnostic results"""
        context = f"""
        PATIENT DIAGNOSTIC CONTEXT:
        - Analysis Type: {diagnostic_context.get('result_type', 'Unknown')}
        - Results: {diagnostic_context.get('prediction', {})}
        - Confidence: {diagnostic_context.get('confidence', 'N/A')}
        - Risk Level: {diagnostic_context.get('risk_level', 'N/A')}
        
        This patient has recently completed a {diagnostic_context.get('result_type', 'medical')} assessment. 
        Please keep this context in mind when providing guidance and recommendations.
        """
        return context
    
    def _format_medical_response(self, response: str) -> str:
        """Format response with medical disclaimers"""
        formatted_response = response
        
        # Add medical disclaimer if not already present
        if "disclaimer" not in response.lower() and "medical professional" not in response.lower():
            disclaimer = """
            
            ⚠️ **IMPORTANT MEDICAL DISCLAIMER:**
            This information is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions. In case of emergency, contact emergency medical services immediately.
            """
            formatted_response += disclaimer
        
        return formatted_response
    
    def _get_fallback_response(self) -> str:
        """Fallback response when AI generation fails"""
        return """I apologize, but I'm experiencing technical difficulties right now. 

For immediate medical concerns, please contact:
- Your healthcare provider
- Emergency services (if urgent)
- A medical helpline

I'll be back online shortly to assist with your health questions.

⚠️ **MEDICAL DISCLAIMER:** This platform provides educational information only and should not replace professional medical advice."""
    
    def _get_fallback_analysis(self) -> str:
        """Fallback analysis when AI generation fails"""
        return """I'm unable to provide a detailed analysis at this moment due to technical issues.

**Important Next Steps:**
1. **Consult a Healthcare Professional:** Share these results with your doctor for proper interpretation
2. **Schedule Follow-up:** Book an appointment for comprehensive evaluation
3. **Keep Records:** Save these results for your medical records
4. **Seek Immediate Care:** If you have concerning symptoms, don't wait

⚠️ **MEDICAL DISCLAIMER:** Diagnostic results require professional medical interpretation. This platform cannot replace clinical expertise."""

# Global chatbot instance
medical_chatbot = MedicalChatbot()

def get_chatbot_response(user_message: str, conversation_history: List[Dict], 
                        diagnostic_context: Optional[Dict] = None, 
                        image_data: Optional[bytes] = None) -> str:
    """Get response from medical chatbot"""
    return medical_chatbot.generate_response(
        user_message, conversation_history, diagnostic_context, image_data
    )

def analyze_diagnostic_with_ai(diagnostic_result: Dict, image_path: Optional[str] = None) -> str:
    """Analyze diagnostic result with AI"""
    return medical_chatbot.analyze_diagnostic_result(diagnostic_result, image_path)