"""
Flask routes for Hygieia 2.0 diagnostic platform
"""
import os
from flask import render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
import logging

from app import app, db
from models import (DiagnosticResult, save_result, get_result, 
                   ChatConversation, ChatMessage, create_chat_conversation, 
                   get_chat_conversation, get_conversations_by_session, add_chat_message)
from ml_models import (
    predict_dermatology, predict_heart_disease, 
    predict_breast_cancer, predict_diabetes
)
from gemini_service import get_educational_content
from image_processing import process_uploaded_image
from chatbot_service import get_chatbot_response, analyze_diagnostic_with_ai
import uuid

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Homepage with diagnostic module selection"""
    return render_template('index.html')

@app.route('/dermatology', methods=['GET', 'POST'])
def dermatology():
    """Dermatological analysis module"""
    if request.method == 'POST':
        try:
            logging.info("Checking for uploaded image in request...")
            if 'image' not in request.files:
                logging.warning("No image file found in request files")
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                logging.warning("Image file selected but filename is empty")
                return jsonify({'error': 'No image selected'}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                logging.info(f"Saving uploaded image: {filename} to {filepath}")
                file.save(filepath)
                
                logging.info(f"Starting image processing for: {filename}")
                processed_image = process_uploaded_image(filepath)
                
                if processed_image is None:
                    logging.error(f"Image processing failed for: {filename}")
                    return jsonify({'error': 'Image processing failed. Please try again.'}), 400
                
                logging.info(f"Image processing completed successfully for: {filename}")
                prediction = predict_dermatology(processed_image)
                
                educational_content = get_educational_content(
                    condition=prediction['condition'],
                    confidence=prediction['confidence'],
                    module_type='dermatology'
                )
                
                result_data = {
                    'result_type': 'dermatology',
                    'input_data': {'image_filename': filename},
                    'prediction': prediction,
                    'educational_content': educational_content,
                    'image_path': filepath
                }
                result_id = save_result(result_data)
                
                return jsonify({'result_id': result_id}), 200  # Return JSON response with result ID
            else:
                logging.warning(f"Invalid file type attempted: {file.filename if file else 'None'}")
                return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or GIF files only.'}), 400
                
        except Exception as e:
            logging.error(f"Error in dermatology analysis: {str(e)}", exc_info=True)
            return jsonify({'error': 'An error occurred during analysis. Please try again.'}), 500

    
    return render_template('dermatology.html')

@app.route('/heart-disease', methods=['GET', 'POST'])
def heart_disease():
    """Heart disease risk assessment"""
    if request.method == 'POST':
        try:
            # Extract form data
            input_data = {
                'age': int(request.form.get('age', 0)),
                'sex': int(request.form.get('sex', 0)),
                'chest_pain_type': int(request.form.get('chest_pain_type', 0)),
                'resting_bp': int(request.form.get('resting_bp', 0)),
                'cholesterol': int(request.form.get('cholesterol', 0)),
                'fasting_blood_sugar': int(request.form.get('fasting_blood_sugar', 0)),
                'resting_ecg': int(request.form.get('resting_ecg', 0)),
                'max_heart_rate': int(request.form.get('max_heart_rate', 0)),
                'exercise_angina': int(request.form.get('exercise_angina', 0)),
                'oldpeak': float(request.form.get('oldpeak', 0.0)),
                'slope': int(request.form.get('slope', 0)),
                'vessels': int(request.form.get('vessels', 0)),
                'thalassemia': int(request.form.get('thalassemia', 0))
            }
            
            # Get prediction
            prediction = predict_heart_disease(input_data)
            
            # Get educational content
            educational_content = get_educational_content(
                condition="Heart Disease Risk Assessment",
                confidence=prediction['confidence'],
                module_type='heart_disease'
            )
            
            # Save result
            result_data = {
                'result_type': 'heart_disease',
                'input_data': input_data,
                'prediction': prediction,
                'educational_content': educational_content
            }
            result_id = save_result(result_data)
            
            return redirect(url_for('results', result_id=result_id))
            
        except Exception as e:
            logging.error(f"Error in heart disease assessment: {str(e)}")
            flash('An error occurred during analysis. Please check your inputs and try again.', 'error')
    
    return render_template('heart_disease.html')

@app.route('/breast-cancer', methods=['GET', 'POST'])
def breast_cancer():
    """Breast cancer risk assessment"""
    if request.method == 'POST':
        try:
            # Extract form data
            input_data = {
                'radius_mean': float(request.form.get('radius_mean', 0.0)),
                'texture_mean': float(request.form.get('texture_mean', 0.0)),
                'perimeter_mean': float(request.form.get('perimeter_mean', 0.0)),
                'area_mean': float(request.form.get('area_mean', 0.0)),
                'smoothness_mean': float(request.form.get('smoothness_mean', 0.0)),
                'compactness_mean': float(request.form.get('compactness_mean', 0.0)),
                'concavity_mean': float(request.form.get('concavity_mean', 0.0)),
                'concave_points_mean': float(request.form.get('concave_points_mean', 0.0)),
                'symmetry_mean': float(request.form.get('symmetry_mean', 0.0)),
                'fractal_dimension_mean': float(request.form.get('fractal_dimension_mean', 0.0))
            }
            
            # Get prediction
            prediction = predict_breast_cancer(input_data)
            
            # Get educational content
            educational_content = get_educational_content(
                condition="Breast Cancer Risk Assessment",
                confidence=prediction['confidence'],
                module_type='breast_cancer'
            )
            
            # Save result
            result_data = {
                'result_type': 'breast_cancer',
                'input_data': input_data,
                'prediction': prediction,
                'educational_content': educational_content
            }
            result_id = save_result(result_data)
            
            return redirect(url_for('results', result_id=result_id))
            
        except Exception as e:
            logging.error(f"Error in breast cancer assessment: {str(e)}")
            flash('An error occurred during analysis. Please check your inputs and try again.', 'error')
    
    return render_template('breast_cancer.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    """Diabetes risk assessment"""
    if request.method == 'POST':
        try:
            # Extract form data
            input_data = {
                'pregnancies': int(request.form.get('pregnancies', 0)),
                'glucose': int(request.form.get('glucose', 0)),
                'blood_pressure': int(request.form.get('blood_pressure', 0)),
                'skin_thickness': int(request.form.get('skin_thickness', 0)),
                'insulin': int(request.form.get('insulin', 0)),
                'bmi': float(request.form.get('bmi', 0.0)),
                'diabetes_pedigree': float(request.form.get('diabetes_pedigree', 0.0)),
                'age': int(request.form.get('age', 0))
            }
            
            # Get prediction
            prediction = predict_diabetes(input_data)
            
            # Get educational content
            educational_content = get_educational_content(
                condition="Diabetes Risk Assessment",
                confidence=prediction['confidence'],
                module_type='diabetes'
            )
            
            # Save result
            result_data = {
                'result_type': 'diabetes',
                'input_data': input_data,
                'prediction': prediction,
                'educational_content': educational_content
            }
            result_id = save_result(result_data)
            
            return redirect(url_for('results', result_id=result_id))
            
        except Exception as e:
            logging.error(f"Error in diabetes assessment: {str(e)}")
            flash('An error occurred during analysis. Please check your inputs and try again.', 'error')
    
    return render_template('diabetes.html')

@app.route('/results/<result_id>')
def results(result_id):
    """Display diagnostic results"""
    result = get_result(result_id)
    if not result:
        flash('Result not found', 'error')
        return redirect(url_for('index'))
    
    return render_template('results.html', result=result)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# Chatbot Routes
@app.route('/chatbot')
def chatbot():
    """Main chatbot interface"""
    # Ensure user has a session ID
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    # Get recent conversations
    conversations = get_conversations_by_session(session['user_id'])
    
    return render_template('chatbot.html', conversations=conversations)

@app.route('/chatbot/conversation/<conversation_id>')
def chatbot_conversation(conversation_id):
    """View specific conversation"""
    conversation = get_chat_conversation(conversation_id)
    if not conversation:
        flash('Conversation not found', 'error')
        return redirect(url_for('chatbot'))
    
    return render_template('chatbot_conversation.html', conversation=conversation)

@app.route('/chatbot/new', methods=['POST'])
def new_chatbot_conversation():
    """Create new chatbot conversation"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    title = request.form.get('title', 'Medical Consultation')
    diagnostic_result_id = request.form.get('diagnostic_result_id')
    
    conversation = create_chat_conversation(
        session_id=session['user_id'],
        title=title,
        diagnostic_result_id=diagnostic_result_id
    )
    
    return redirect(url_for('chatbot_conversation', conversation_id=conversation.id))

@app.route('/chatbot/message', methods=['POST'])
def send_chatbot_message():
    """Send message to chatbot"""
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        user_message = data.get('message')
        
        if not conversation_id or not user_message:
            return jsonify({'error': 'Missing required fields'}), 400
        
        conversation = get_chat_conversation(conversation_id)
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        # Add user message
        add_chat_message(conversation_id, 'user', user_message)
        
        # Prepare conversation history
        conversation_history = []
        for msg in conversation.messages:
            conversation_history.append({
                'role': msg.role,
                'content': msg.content
            })
        
        # Get diagnostic context if available
        diagnostic_context = None
        if conversation.diagnostic_result:
            diagnostic_context = {
                'result_type': conversation.diagnostic_result.result_type,
                'prediction': conversation.diagnostic_result.prediction,
                'confidence': conversation.diagnostic_result.prediction.get('confidence'),
                'risk_level': conversation.diagnostic_result.prediction.get('risk_level')
            }
        
        # Get AI response
        ai_response = get_chatbot_response(
            user_message, 
            conversation_history[:-1],  # Exclude the just-added user message
            diagnostic_context
        )
        
        # Add AI response
        ai_message = add_chat_message(conversation_id, 'assistant', ai_response, 'text')
        
        return jsonify({
            'success': True,
            'message': {
                'id': ai_message.id,
                'role': ai_message.role,
                'content': ai_message.content,
                'created_at': ai_message.created_at.isoformat()
            }
        })
        
    except Exception as e:
        logging.error(f"Error in chatbot message: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/chatbot/analyze/<result_id>')
def analyze_result_with_chatbot(result_id):
    """Start chatbot conversation with diagnostic result analysis"""
    result = get_result(result_id)
    if not result:
        flash('Diagnostic result not found', 'error')
        return redirect(url_for('chatbot'))
    
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    # Create conversation with diagnostic context
    title = f"Analysis: {result.result_type.replace('_', ' ').title()} Results"
    conversation = create_chat_conversation(
        session_id=session['user_id'],
        title=title,
        diagnostic_result_id=result.id
    )
    
    # Generate initial AI analysis
    result_dict = {
        'result_type': result.result_type,
        'prediction': result.prediction,
        'input_data': result.input_data
    }
    
    ai_analysis = analyze_diagnostic_with_ai(result_dict, result.image_path)
    
    # Add initial analysis message
    add_chat_message(
        conversation.id, 
        'assistant', 
        f"I've analyzed your {result.result_type.replace('_', ' ')} results. Here's my detailed assessment:\n\n{ai_analysis}",
        'image_analysis'
    )
    
    return redirect(url_for('chatbot_conversation', conversation_id=conversation.id))
