"""
Google Gemini API integration for patient education content generation.
"""
import os
import logging
from google import genai
from google.genai import types

# Initialize Gemini client with error handling
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = None

if gemini_api_key and gemini_api_key != "your_gemini_api_key_here":
    try:
        client = genai.Client(api_key=gemini_api_key)
        print("Gemini API client initialized successfully")
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        client = None
else:
    print("Gemini API key not set. Using fallback educational content only.")

def get_educational_content(condition: str, confidence: float, module_type: str) -> str:
    """
    Generate patient-friendly educational content using Google Gemini.
    
    Args:
        condition: The predicted condition or risk level
        confidence: Confidence score of the prediction
        module_type: Type of diagnostic module (dermatology, heart_disease, etc.)
    
    Returns:
        Educational content string with medical disclaimer
    """
    # Use fallback content if Gemini client is not available
    if client is None:
        return get_fallback_content(condition, module_type)
    
    try:
        # Create educational prompt based on module type
        system_prompt = """You are a medical education specialist. Your task is to provide clear, 
        patient-friendly educational information about medical conditions. Your response should be:
        
        1. Educational and informative, not diagnostic
        2. Written in simple, understandable language
        3. Focused on general information about the condition
        4. Include lifestyle recommendations and prevention tips
        5. Emphasize the importance of professional medical consultation
        6. Be empathetic and supportive in tone
        
        IMPORTANT: Always emphasize that this is educational information only and should not 
        replace professional medical advice, diagnosis, or treatment."""
        
        # Create condition-specific prompt
        if module_type == 'dermatology':
            content_prompt = f"""
            Provide educational information about {condition}, a skin condition that was identified 
            with {confidence*100:.1f}% confidence by our analysis system. Explain:
            
            - What this condition typically involves
            - Common characteristics and symptoms
            - General treatment approaches
            - When to seek medical attention
            - Prevention and skin care tips
            
            Keep the tone reassuring but emphasize the need for professional dermatological evaluation.
            """
        
        elif module_type == 'heart_disease':
            content_prompt = f"""
            Provide educational information about heart disease risk assessment. Our analysis indicates 
            a {condition} level with {confidence*100:.1f}% confidence. Explain:
            
            - What heart disease risk factors mean
            - Lifestyle factors that influence heart health
            - General prevention strategies
            - When to consult a cardiologist
            - Heart-healthy lifestyle recommendations
            
            Emphasize that risk assessment is not a diagnosis and professional evaluation is essential.
            """
        
        elif module_type == 'breast_cancer':
            content_prompt = f"""
            Provide educational information about breast health and the {condition} classification 
            identified with {confidence*100:.1f}% confidence. Explain:
            
            - What this classification typically means
            - The importance of professional medical evaluation
            - General information about breast health
            - Follow-up care recommendations
            - Emotional support and coping strategies
            
            Be sensitive and supportive while emphasizing the critical need for professional medical consultation.
            """
        
        elif module_type == 'diabetes':
            content_prompt = f"""
            Provide educational information about diabetes risk assessment. Our analysis indicates 
            {condition} with {confidence*100:.1f}% confidence. Explain:
            
            - What diabetes risk levels mean
            - Lifestyle factors that influence diabetes risk
            - Prevention strategies and healthy habits
            - When to seek medical evaluation
            - Diet and exercise recommendations
            
            Emphasize that this is a risk assessment, not a diagnosis, and professional medical evaluation is needed.
            """
        
        else:
            content_prompt = f"""
            Provide general health education information about {condition} identified with 
            {confidence*100:.1f}% confidence by our analysis system. Focus on general health 
            maintenance and the importance of professional medical consultation.
            """
        
        # Generate content using Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[types.Part(text=content_prompt)])
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,  # Lower temperature for more consistent medical content
                max_output_tokens=1000
            )
        )
        
        if response.text:
            # Add mandatory medical disclaimer
            educational_content = response.text
            disclaimer = """
            
            ⚠️ IMPORTANT MEDICAL DISCLAIMER:
            This information is for educational purposes only and should not be considered as 
            medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
            professionals for proper medical evaluation, diagnosis, and treatment decisions. 
            Do not rely solely on this automated analysis for medical decisions.
            """
            
            return educational_content + disclaimer
        else:
            return get_fallback_content(condition, module_type)
            
    except Exception as e:
        logging.error(f"Error generating educational content with Gemini: {str(e)}")
        return get_fallback_content(condition, module_type)

def get_fallback_content(condition: str, module_type: str) -> str:
    """
    Fallback educational content when Gemini API is unavailable.
    """
    fallback_content = {
        'dermatology': f"""
        Educational Information: {condition}
        
        Skin conditions can vary widely in their appearance and significance. It's important to 
        monitor any changes in your skin and seek professional evaluation from a dermatologist 
        for accurate diagnosis and appropriate treatment.
        
        General skin health tips:
        • Protect your skin from sun exposure
        • Use appropriate sunscreen
        • Perform regular skin self-examinations
        • Maintain good skin hygiene
        • Stay hydrated and eat a balanced diet
        
        ⚠️ IMPORTANT: This automated analysis is not a medical diagnosis. Please consult 
        a qualified dermatologist for proper evaluation and treatment recommendations.
        """,
        
        'heart_disease': f"""
        Heart Health Information: {condition}
        
        Understanding your heart disease risk can help you make informed decisions about your health. 
        Risk factors include age, cholesterol levels, blood pressure, and lifestyle factors.
        
        Heart-healthy recommendations:
        • Maintain a balanced, low-sodium diet
        • Exercise regularly (as approved by your doctor)
        • Don't smoke and limit alcohol consumption
        • Manage stress effectively
        • Monitor blood pressure and cholesterol
        
        ⚠️ IMPORTANT: This risk assessment is not a medical diagnosis. Consult with a 
        cardiologist or your primary care physician for comprehensive evaluation.
        """,
        
        'breast_cancer': f"""
        Breast Health Information: {condition}
        
        Breast health is an important aspect of overall wellness. Regular monitoring and 
        professional evaluation are essential for maintaining breast health.
        
        General recommendations:
        • Perform regular breast self-examinations
        • Follow mammogram screening guidelines
        • Maintain a healthy lifestyle
        • Know your family history
        • Report any changes to your healthcare provider immediately
        
        ⚠️ CRITICAL: This analysis requires immediate professional medical evaluation. 
        Please consult with an oncologist or breast specialist without delay.
        """,
        
        'diabetes': f"""
        Diabetes Risk Information: {condition}
        
        Understanding diabetes risk factors can help you take preventive measures and maintain 
        better health through lifestyle choices.
        
        Diabetes prevention tips:
        • Maintain a healthy weight
        • Follow a balanced diet low in processed sugars
        • Exercise regularly
        • Monitor blood sugar levels as recommended
        • Stay hydrated and get adequate sleep
        
        ⚠️ IMPORTANT: This risk assessment is educational only and not a medical diagnosis. 
        Consult with an endocrinologist or primary care physician for proper evaluation.
        """
    }
    
    return fallback_content.get(module_type, f"""
    Educational Information: {condition}
    
    This analysis provides general health information. For accurate diagnosis and treatment, 
    please consult with qualified healthcare professionals.
    
    ⚠️ IMPORTANT MEDICAL DISCLAIMER:
    This information is for educational purposes only and should not replace professional 
    medical advice, diagnosis, or treatment.
    """)
