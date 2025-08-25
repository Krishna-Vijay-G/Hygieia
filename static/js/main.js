/**
 * Hygieia 2.0 - Main JavaScript File
 * Enhanced user interactions and medical platform functionality
 */

// Global application state
window.HygieiaApp = {
    isProcessing: false,
    currentModule: null,
    uploadedFiles: {},
    formValidation: {},
    
    // Initialize application
    init: function() {
        this.setupEventListeners();
        this.initializeTooltips();
        this.setupFormValidation();
        this.setupImageHandling();
        this.setupAccessibility();
        this.detectCurrentModule();
        console.log('Hygieia 2.0 initialized successfully');
    },
    
    // Detect current module based on URL
    detectCurrentModule: function() {
        const path = window.location.pathname;
        if (path.includes('dermatology')) this.currentModule = 'dermatology';
        else if (path.includes('heart-disease')) this.currentModule = 'heart_disease';
        else if (path.includes('breast-cancer')) this.currentModule = 'breast_cancer';
        else if (path.includes('diabetes')) this.currentModule = 'diabetes';
        else if (path.includes('results')) this.currentModule = 'results';
        else this.currentModule = 'home';
    }
};

// Event Listeners Setup
HygieiaApp.setupEventListeners = function() {
    // Form submission handlers
    document.addEventListener('submit', this.handleFormSubmission.bind(this));
    
    // File upload handlers
    document.addEventListener('change', this.handleFileSelection.bind(this));
    
    // Navigation enhancement
    document.addEventListener('click', this.handleNavigation.bind(this));
    
    window.addEventListener('beforeunload', this.handleBeforeUnload.bind(this));
    window.addEventListener('resize', this.handleResize.bind(this));
    
    // Medical disclaimer acknowledgment
    this.setupDisclaimerHandling();
};

// Initialize Bootstrap tooltips
HygieiaApp.initializeTooltips = function() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
};

// Form Validation Setup
HygieiaApp.setupFormValidation = function() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const formId = form.id || 'unknown';
        this.formValidation[formId] = {
            isValid: false,
            errors: [],
            requiredFields: form.querySelectorAll('[required]')
        };
        
        // Real-time validation
        form.addEventListener('input', this.validateFormField.bind(this));
        form.addEventListener('blur', this.validateFormField.bind(this), true);
    });
};

// Image Handling Setup
HygieiaApp.setupImageHandling = function() {
    // Drag and drop for image uploads
    const dropZones = document.querySelectorAll('.image-upload-area, [type="file"]');
    dropZones.forEach(zone => {
        zone.addEventListener('dragover', this.handleDragOver.bind(this));
        zone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        zone.addEventListener('drop', this.handleFileDrop.bind(this));
    });
};

// Accessibility Setup
HygieiaApp.setupAccessibility = function() {
    // Skip to main content link
    this.createSkipLink();
    
    // Keyboard navigation for cards
    const cards = document.querySelectorAll('.diagnostic-card');
    cards.forEach(card => {
        card.setAttribute('tabindex', '0');
        card.setAttribute('role', 'button');
        card.addEventListener('keydown', this.handleCardKeydown.bind(this));
    });
    
    // Screen reader announcements for dynamic content
    this.createAriaLiveRegion();
};

// Form Submission Handler
HygieiaApp.handleFormSubmission = function(event) {
    const form = event.target.closest('form');
    if (!form) return;
    
    // Prevent multiple submissions
    if (this.isProcessing) {
        event.preventDefault();
        return;
    }
    
    // Validate form before submission
    const isValid = this.validateForm(form);
    if (!isValid) {
        event.preventDefault();
        this.showValidationErrors(form);
        return;
    }
    
    // Medical disclaimer confirmation for sensitive modules
    if (this.currentModule === 'breast_cancer' || this.currentModule === 'heart_disease') {
        if (!confirm('This analysis is for educational purposes only and requires professional medical validation. Do you want to continue?')) {
            event.preventDefault();
            return;
        }
    }
    
    this.isProcessing = true;
    this.showProcessingState(form);
    
    // Add loading analytics
    this.trackAnalytics('form_submission', {
        module: this.currentModule,
        timestamp: new Date().toISOString()
    });
};

// File Selection Handler
HygieiaApp.handleFileSelection = function(event) {
    const input = event.target;
    if (input.type !== 'file') return;
    
    const files = Array.from(input.files);
    files.forEach(file => {
        this.validateMedicalImage(file, input);
    });
};

// Medical Image Validation
HygieiaApp.validateMedicalImage = function(file, inputElement) {
    const validation = {
        valid: true,
        errors: [],
        warnings: []
    };
    
    // File size validation (16MB limit)
    if (file.size > 16 * 1024 * 1024) {
        validation.valid = false;
        validation.errors.push('File size exceeds 16MB limit');
    }
    
    // File type validation
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
    if (!allowedTypes.includes(file.type)) {
        validation.valid = false;
        validation.errors.push('Invalid file type. Please upload JPG, PNG, or GIF files only');
    }
    
    // Image dimension validation
    this.validateImageDimensions(file).then(dimensions => {
        if (dimensions.width < 100 || dimensions.height < 100) {
            validation.warnings.push('Image resolution may be too low for accurate analysis');
        }
        
        if (dimensions.width > 4000 || dimensions.height > 4000) {
            validation.warnings.push('High resolution image detected. Processing may take longer');
        }
        
        this.showImageValidationResult(validation, inputElement);
    });
    
    return validation.valid;
};

// Validate Image Dimensions
HygieiaApp.validateImageDimensions = function(file) {
    return new Promise((resolve) => {
        const img = new Image();
        const url = URL.createObjectURL(file);
        
        img.onload = function() {
            resolve({
                width: this.naturalWidth,
                height: this.naturalHeight
            });
            URL.revokeObjectURL(url);
        };
        
        img.onerror = function() {
            resolve({ width: 0, height: 0 });
            URL.revokeObjectURL(url);
        };
        
        img.src = url;
    });
};

// Form Validation
HygieiaApp.validateForm = function(form) {
    const formId = form.id || 'unknown';
    const validation = this.formValidation[formId];
    
    if (!validation) return true;
    
    validation.errors = [];
    validation.isValid = true;
    
    // Validate required fields
    validation.requiredFields.forEach(field => {
        if (!this.validateRequiredField(field)) {
            validation.isValid = false;
            validation.errors.push(`${this.getFieldLabel(field)} is required`);
        }
    });
    
    // Module-specific validations
    if (this.currentModule === 'heart_disease') {
        this.validateHeartDiseaseForm(form, validation);
    } else if (this.currentModule === 'breast_cancer') {
        this.validateBreastCancerForm(form, validation);
    } else if (this.currentModule === 'diabetes') {
        this.validateDiabetesForm(form, validation);
    }
    
    return validation.isValid;
};

// Field Validation
HygieiaApp.validateFormField = function(event) {
    const field = event.target;
    const fieldType = field.type || field.tagName.toLowerCase();
    
    // Clear previous validation state
    field.classList.remove('is-invalid', 'is-valid');
    
    let isValid = true;
    let errorMessage = '';
    
    // Required field validation
    if (field.hasAttribute('required') && !this.validateRequiredField(field)) {
        isValid = false;
        errorMessage = `${this.getFieldLabel(field)} is required`;
    }
    
    // Type-specific validations
    if (isValid && field.value) {
        switch (fieldType) {
            case 'number':
                isValid = this.validateNumericField(field);
                break;
            case 'email':
                isValid = this.validateEmailField(field);
                break;
            case 'file':
                // File validation handled separately
                break;
        }
    }
    
    // Medical field range validations
    if (isValid && field.value) {
        isValid = this.validateMedicalRange(field);
    }
    
    // Apply validation styling
    if (isValid) {
        field.classList.add('is-valid');
    } else {
        field.classList.add('is-invalid');
        this.showFieldError(field, errorMessage);
    }
    
    return isValid;
};

// Medical Range Validation
HygieiaApp.validateMedicalRange = function(field) {
    const fieldName = field.name;
    const value = parseFloat(field.value);
    
    if (isNaN(value)) return true;
    
    const medicalRanges = {
        // Heart disease ranges
        'age': { min: 18, max: 120, warning: { min: 65, max: 120 } },
        'resting_bp': { min: 80, max: 250, warning: { min: 140, max: 250 } },
        'cholesterol': { min: 100, max: 600, warning: { min: 240, max: 600 } },
        'max_heart_rate': { min: 60, max: 220, warning: { min: 60, max: 100 } },
        
        // Diabetes ranges
        'glucose': { min: 50, max: 300, warning: { min: 140, max: 300 } },
        'bmi': { min: 10, max: 70, warning: { min: 25, max: 70 } },
        'blood_pressure': { min: 40, max: 130, warning: { min: 90, max: 130 } },
        
        // Breast cancer ranges
        'radius_mean': { min: 0, max: 50, typical: { min: 10, max: 20 } },
        'texture_mean': { min: 0, max: 50, typical: { min: 10, max: 25 } },
        'area_mean': { min: 0, max: 3000, typical: { min: 350, max: 1200 } }
    };
    
    const range = medicalRanges[fieldName];
    if (!range) return true;
    
    // Check basic range
    if (value < range.min || value > range.max) {
        this.showFieldWarning(field, `Value should be between ${range.min} and ${range.max}`);
        return false;
    }
    
    // Check warning ranges
    if (range.warning && (value >= range.warning.min && value <= range.warning.max)) {
        this.showFieldWarning(field, 'This value may indicate increased health risk');
    }
    
    return true;
};

// Processing State Display
HygieiaApp.showProcessingState = function(form) {
    const submitButton = form.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
    }
    
    // Show processing modal if it exists
    const processingModal = document.getElementById('processingModal');
    if (processingModal) {
        const modal = new bootstrap.Modal(processingModal);
        modal.show();
        
        // Animate progress bar
        this.animateProgressBar();
    }
    
    // Announce to screen readers
    this.announceToScreenReader('Processing your request. Please wait...');
};

// Progress Bar Animation
HygieiaApp.animateProgressBar = function() {
    const progressBar = document.querySelector('.progress-bar');
    if (!progressBar) return;
    
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) {
            progress = 90;
            clearInterval(interval);
        }
        progressBar.style.width = progress + '%';
        progressBar.setAttribute('aria-valuenow', Math.round(progress));
    }, 300);
};

// Drag and Drop Handlers
HygieiaApp.handleDragOver = function(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.add('dragover');
};

HygieiaApp.handleDragLeave = function(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('dragover');
};

HygieiaApp.handleFileDrop = function(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('dragover');
    
    const files = Array.from(event.dataTransfer.files);
    const fileInput = document.querySelector('input[type="file"]');
    
    if (fileInput && files.length > 0) {
        // Create new FileList
        const dt = new DataTransfer();
        files.forEach(file => dt.items.add(file));
        fileInput.files = dt.files;
        
        // Trigger change event
        fileInput.dispatchEvent(new Event('change', { bubbles: true }));
        
        // Preview first image
        if (files[0] && files[0].type.startsWith('image/')) {
            this.previewImage(files[0]);
        }
    }
};

// Image Preview
HygieiaApp.previewImage = function(file) {
    const preview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    
    if (!preview || !previewImg) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        previewImg.alt = `Preview of ${file.name}`;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
};

// Navigation Enhancement
HygieiaApp.handleNavigation = function(event) {
    const link = event.target.closest('a');
    if (!link) return;
    
    // Smooth scrolling for internal links
    if (link.hash && link.getAttribute('href').startsWith('#')) {
        event.preventDefault();
        this.smoothScrollTo(link.hash);
    }
    
    // Analytics tracking
    this.trackAnalytics('navigation_click', {
        href: link.getAttribute('href'),
        text: link.textContent.trim()
    });
};

// Card Keyboard Navigation
HygieiaApp.handleCardKeydown = function(event) {
    const card = event.currentTarget;
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        const link = card.querySelector('a');
        if (link) link.click();
    }
};

// Medical Disclaimer Handling
HygieiaApp.setupDisclaimerHandling = function() {
    const disclaimerBanner = document.querySelector('.alert-warning[role="alert"]');
    if (disclaimerBanner) {
        // Track disclaimer views
        this.trackAnalytics('disclaimer_viewed', {
            module: this.currentModule,
            timestamp: new Date().toISOString()
        });
        
        // Auto-hide disclaimer after extended viewing
        setTimeout(() => {
            const closeButton = disclaimerBanner.querySelector('.btn-close');
            if (closeButton && disclaimerBanner.offsetParent) {
                closeButton.style.animation = 'pulse 1s infinite';
            }
        }, 30000);
    }
};

// Utility Functions
HygieiaApp.validateRequiredField = function(field) {
    if (field.type === 'file') {
        return field.files && field.files.length > 0;
    }
    return field.value && field.value.trim() !== '';
};

HygieiaApp.validateNumericField = function(field) {
    const value = parseFloat(field.value);
    const min = parseFloat(field.min);
    const max = parseFloat(field.max);
    
    if (isNaN(value)) return false;
    if (!isNaN(min) && value < min) return false;
    if (!isNaN(max) && value > max) return false;
    
    return true;
};

HygieiaApp.validateEmailField = function(field) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(field.value);
};

HygieiaApp.getFieldLabel = function(field) {
    const label = document.querySelector(`label[for="${field.id}"]`);
    return label ? label.textContent.replace('*', '').trim() : field.name || 'Field';
};

HygieiaApp.showFieldError = function(field, message) {
    let errorDiv = field.parentElement.querySelector('.invalid-feedback');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.className = 'invalid-feedback';
        field.parentElement.appendChild(errorDiv);
    }
    errorDiv.textContent = message;
};

HygieiaApp.showFieldWarning = function(field, message) {
    let warningDiv = field.parentElement.querySelector('.warning-feedback');
    if (!warningDiv) {
        warningDiv = document.createElement('div');
        warningDiv.className = 'warning-feedback text-warning small';
        field.parentElement.appendChild(warningDiv);
    }
    warningDiv.innerHTML = `<i class="fas fa-exclamation-triangle me-1"></i>${message}`;
};

HygieiaApp.showValidationErrors = function(form) {
    const formId = form.id || 'unknown';
    const validation = this.formValidation[formId];
    
    if (validation && validation.errors.length > 0) {
        const errorMessage = validation.errors.join('\n');
        alert('Please correct the following errors:\n\n' + errorMessage);
        
        // Focus first invalid field
        const firstInvalid = form.querySelector('.is-invalid');
        if (firstInvalid) {
            firstInvalid.focus();
            firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
};

HygieiaApp.smoothScrollTo = function(target) {
    const element = document.querySelector(target);
    if (element) {
        element.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }
};

HygieiaApp.createSkipLink = function() {
    const skipLink = document.createElement('a');
    skipLink.href = '#main-content';
    skipLink.textContent = 'Skip to main content';
    skipLink.className = 'sr-only sr-only-focusable btn btn-primary position-absolute';
    skipLink.style.zIndex = '9999';
    skipLink.style.top = '10px';
    skipLink.style.left = '10px';
    
    document.body.insertBefore(skipLink, document.body.firstChild);
};

HygieiaApp.createAriaLiveRegion = function() {
    const liveRegion = document.createElement('div');
    liveRegion.id = 'aria-live-region';
    liveRegion.className = 'sr-only';
    liveRegion.setAttribute('aria-live', 'polite');
    liveRegion.setAttribute('aria-atomic', 'true');
    
    document.body.appendChild(liveRegion);
};

HygieiaApp.announceToScreenReader = function(message) {
    const liveRegion = document.getElementById('aria-live-region');
    if (liveRegion) {
        liveRegion.textContent = message;
        
        // Clear after announcement
        setTimeout(() => {
            liveRegion.textContent = '';
        }, 1000);
    }
};

HygieiaApp.trackAnalytics = function(event, data) {
    // Analytics tracking (placeholder for actual implementation)
    console.log('Analytics Event:', event, data);
    
    // Example integration with Google Analytics
    if (typeof gtag !== 'undefined') {
        gtag('event', event, {
            custom_parameter_1: data.module || '',
            custom_parameter_2: data.timestamp || ''
        });
    }
};

HygieiaApp.handleBeforeUnload = function(event) {
    // Only show leave warning for actual form submissions in progress
    // Don't show for normal navigation or when processing is complete
    if (this.isProcessing) {
        event.preventDefault();
        event.returnValue = 'You have an analysis in progress. Are you sure you want to leave?';
        return event.returnValue;
    }
};

HygieiaApp.handleResize = function() {
    // Responsive adjustments
    const isMobile = window.innerWidth < 768;
    document.body.classList.toggle('mobile-view', isMobile);
};

// Module-specific validations
HygieiaApp.validateHeartDiseaseForm = function(form, validation) {
    const age = parseInt(form.querySelector('[name="age"]')?.value || 0);
    const cholesterol = parseInt(form.querySelector('[name="cholesterol"]')?.value || 0);
    const restingBp = parseInt(form.querySelector('[name="resting_bp"]')?.value || 0);
    
    if (age > 100) {
        validation.warnings = validation.warnings || [];
        validation.warnings.push('Very advanced age may affect result accuracy');
    }
    
    if (cholesterol > 400) {
        validation.warnings = validation.warnings || [];
        validation.warnings.push('Extremely high cholesterol level detected');
    }
    
    if (restingBp > 180) {
        validation.warnings = validation.warnings || [];
        validation.warnings.push('Hypertensive crisis range - seek immediate medical attention');
    }
};

HygieiaApp.validateBreastCancerForm = function(form, validation) {
    const radiusMean = parseFloat(form.querySelector('[name="radius_mean"]')?.value || 0);
    const areaMean = parseFloat(form.querySelector('[name="area_mean"]')?.value || 0);
    
    if (radiusMean > 25) {
        validation.warnings = validation.warnings || [];
        validation.warnings.push('Large radius measurement detected');
    }
    
    if (areaMean > 2000) {
        validation.warnings = validation.warnings || [];
        validation.warnings.push('Large area measurement detected');
    }
};

HygieiaApp.validateDiabetesForm = function(form, validation) {
    const glucose = parseInt(form.querySelector('[name="glucose"]')?.value || 0);
    const bmi = parseFloat(form.querySelector('[name="bmi"]')?.value || 0);
    
    if (glucose >= 200) {
        validation.warnings = validation.warnings || [];
        validation.warnings.push('Glucose level in diabetic range - immediate medical consultation recommended');
    }
    
    if (bmi >= 35) {
        validation.warnings = validation.warnings || [];
        validation.warnings.push('Severely obese BMI - high diabetes risk');
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    HygieiaApp.init();
});

// Export for testing purposes
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HygieiaApp;
}
