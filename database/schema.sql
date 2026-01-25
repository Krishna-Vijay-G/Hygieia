-- ============================================================
-- Hygieia Medical Diagnostic Platform
-- PostgreSQL Database Schema for Supabase
-- ============================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- USERS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    first_name VARCHAR(80) NOT NULL,
    last_name VARCHAR(80) NOT NULL,
    phone VARCHAR(20),
    avatar_url VARCHAR(500),
    is_admin BOOLEAN DEFAULT FALSE NOT NULL,
    is_owner BOOLEAN DEFAULT FALSE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    is_verified BOOLEAN DEFAULT FALSE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

-- Indexes for users
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users(is_admin);
CREATE INDEX IF NOT EXISTS idx_users_is_owner ON users(is_owner);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- ============================================================
-- ANALYSES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,
    model_name VARCHAR(100),
    input_data JSONB,
    result JSONB NOT NULL,
    risk_level VARCHAR(50),
    confidence DOUBLE PRECISION,
    risk_score DOUBLE PRECISION,
    image_path VARCHAR(500),
    blockchain_hash VARCHAR(256),
    ai_summary TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Indexes for analyses
CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_analyses_type ON analyses(analysis_type);
CREATE INDEX IF NOT EXISTS idx_analyses_blockchain ON analyses(blockchain_hash);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);

-- Comment for ai_summary column
COMMENT ON COLUMN analyses.ai_summary IS 'Auto-generated AI summary stored permanently after analysis creation. Generated automatically when analysis is first created and persists on page reloads.';

-- ============================================================
-- BLOCKCHAIN RECORDS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS blockchain_records (
    id SERIAL PRIMARY KEY,
    block_index INTEGER UNIQUE NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    data_hash VARCHAR(256) NOT NULL,
    previous_hash VARCHAR(256) NOT NULL,
    current_hash VARCHAR(256) UNIQUE NOT NULL,
    nonce INTEGER DEFAULT 0,
    analysis_id UUID REFERENCES analyses(id) ON DELETE SET NULL
);

-- Indexes for blockchain
CREATE INDEX IF NOT EXISTS idx_blockchain_index ON blockchain_records(block_index);
CREATE INDEX IF NOT EXISTS idx_blockchain_hash ON blockchain_records(current_hash);
CREATE INDEX IF NOT EXISTS idx_blockchain_data_hash ON blockchain_records(data_hash);
CREATE INDEX IF NOT EXISTS idx_blockchain_analysis ON blockchain_records(analysis_id);

-- ============================================================
-- MODEL BENCHMARKS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS model_benchmarks (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    accuracy DOUBLE PRECISION,
    precision_score DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    auc_roc DOUBLE PRECISION,
    test_samples INTEGER,
    training_samples INTEGER,
    benchmark_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    additional_metrics JSONB
);

-- Index for benchmarks
CREATE INDEX IF NOT EXISTS idx_benchmarks_model_type ON model_benchmarks(model_type);

-- ============================================================
-- AUDIT LOGS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(36),
    details JSONB,
    ip_address VARCHAR(45),
    user_agent VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Indexes for audit logs
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at);

-- ============================================================
-- TRIGGERS AND FUNCTIONS
-- ============================================================

-- Function to ensure username is stored in lowercase
CREATE OR REPLACE FUNCTION lowercase_username()
RETURNS TRIGGER AS $$
BEGIN
    NEW.username = LOWER(NEW.username);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically convert username to lowercase on insert/update
CREATE TRIGGER trigger_lowercase_username
    BEFORE INSERT OR UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION lowercase_username();

-- ============================================================
-- INITIAL DATA - Create Default Admin User
-- Password: Admin@123 (change immediately after setup)
-- ============================================================
INSERT INTO users (
    id,
    username,
    email,
    password_hash,
    first_name,
    last_name,
    is_admin,
    is_owner,
    is_active,
    is_verified
) VALUES (
    uuid_generate_v4(),
    'hygieia',
    'hygieia.gkv@gmail.com',
    -- This is bcrypt hash for 'Hygieia@hygieia11' - CHANGE THIS PASSWORD IMMEDIATELY
    'scrypt:32768:8:1$f8iuYr7kDlAlBOJB$57f7d4f083b48e399fb6d90e8da0df29e9d1bdf93da50c226eafefda287f136b8d2fc3e9f3d4ddf795e2fc8c640a80a73ff97be5c83da19e4cc7b4a55020a238',
    'Hygieia',
    'Administrator',
    TRUE,
    TRUE,
    TRUE,
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- ============================================================
-- INITIAL BENCHMARK DATA
-- ============================================================
INSERT INTO model_benchmarks (model_type, model_name, accuracy, precision_score, recall, f1_score, auc_roc, test_samples, training_samples, additional_metrics)
VALUES
    ('heart-prediction', 'Heart Risk Predictive Model', 0.994, 0.993, 0.995, 0.994, 0.999, 14000, 56000, '{"features": 18, "algorithm": "AdaBoost"}'),
    ('diabetes-prediction', 'Diabetes Risk Predictive Model', 0.981, 0.979, 0.983, 0.981, 0.995, 104, 416, '{"features": 16, "algorithm": "LightGBM"}'),
    ('skin-diagnosis', 'Skin Lesion Diagnostic Model', 0.968, 0.965, 0.970, 0.967, 0.985, 4600, 18400, '{"classes": 23, "algorithm": "CNN + Transfer Learning"}'),
    ('breast-prediction', 'BC Predictive Model', 0.813, 0.810, 0.813, 0.811, 0.902, 5000, 20000, '{"features": 10, "algorithm": "LightGBM"}'),
    ('breast-diagnosis', 'BC Diagnostic Model', 0.972, 0.974, 0.968, 0.971, 0.994, 114, 455, '{"features": 30, "algorithm": "Random Forest"}')
ON CONFLICT DO NOTHING;

-- ============================================================
-- ROW LEVEL SECURITY (RLS) POLICIES FOR SUPABASE
-- ============================================================

-- Enable RLS
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE blockchain_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Users can read their own data
CREATE POLICY "Users can view own profile" ON users
    FOR SELECT USING (auth.uid()::text = id::text);

-- Users can update their own profile
CREATE POLICY "Users can update own profile" ON users
    FOR UPDATE USING (auth.uid()::text = id::text);

-- Users can view their own analyses
CREATE POLICY "Users can view own analyses" ON analyses
    FOR SELECT USING (auth.uid()::text = user_id::text);

-- Users can insert their own analyses
CREATE POLICY "Users can create analyses" ON analyses
    FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

-- Admins can view all data (bypass RLS with service role)

-- ============================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for users table
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to validate blockchain integrity
CREATE OR REPLACE FUNCTION validate_blockchain()
RETURNS TABLE (
    is_valid BOOLEAN,
    invalid_blocks INTEGER[],
    total_blocks INTEGER
) AS $$
DECLARE
    prev_hash VARCHAR(256);
    block RECORD;
    invalid INTEGER[] := '{}';
BEGIN
    total_blocks := 0;
    is_valid := TRUE;
    prev_hash := REPEAT('0', 64);
    
    FOR block IN SELECT * FROM blockchain_records ORDER BY block_index ASC
    LOOP
        total_blocks := total_blocks + 1;
        IF block.previous_hash != prev_hash THEN
            is_valid := FALSE;
            invalid := array_append(invalid, block.block_index);
        END IF;
        prev_hash := block.current_hash;
    END LOOP;
    
    invalid_blocks := invalid;
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================

-- User analysis summary view
CREATE OR REPLACE VIEW user_analysis_summary AS
SELECT 
    u.id AS user_id,
    u.username,
    u.first_name || ' ' || u.last_name AS full_name,
    COUNT(a.id) AS total_analyses,

    COUNT(CASE WHEN a.analysis_type = 'heart-prediction' THEN 1 END) AS heart_prediction_analyses,
    COUNT(CASE WHEN a.analysis_type = 'diabetes-prediction' THEN 1 END) AS diabetes_prediction_analyses,
    COUNT(CASE WHEN a.analysis_type = 'skin-diagnosis' THEN 1 END) AS skin_diagnosis_analyses,
    COUNT(CASE WHEN a.analysis_type = 'breast-prediction' THEN 1 END) AS breast_prediction_analyses,
    COUNT(CASE WHEN a.analysis_type = 'breast-diagnosis' THEN 1 END) AS breast_diagnosis_analyses,
    MAX(a.created_at) AS last_analysis_date
FROM users u
LEFT JOIN analyses a ON u.id = a.user_id
GROUP BY u.id, u.username, u.first_name, u.last_name;

-- Blockchain overview view
CREATE OR REPLACE VIEW blockchain_overview AS
SELECT 
    br.block_index,
    br.timestamp,
    br.current_hash,
    a.analysis_type,
    a.risk_level,
    u.username,
    u.first_name || ' ' || u.last_name AS user_full_name
FROM blockchain_records br
LEFT JOIN analyses a ON br.analysis_id = a.id
LEFT JOIN users u ON a.user_id = u.id
ORDER BY br.block_index DESC;

-- ============================================================
-- GRANT PERMISSIONS (adjust as needed)
-- ============================================================
-- These are typically handled by Supabase dashboard
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO authenticated;


--give the codes to add the 2 diff breeast to the user analysis summary

-- ============================================================
-- CHAT SESSIONS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) DEFAULT 'New Conversation',
    analysis_id UUID REFERENCES analyses(id) ON DELETE SET NULL,
    context_type VARCHAR(50),  -- 'general', 'analysis', 'follow-up'
    context_data JSONB,
    messages JSONB DEFAULT '[]'::jsonb NOT NULL,  -- Store all messages as JSON array
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for chat sessions
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_analysis_id ON chat_sessions(analysis_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_created_at ON chat_sessions(created_at);

-- Trigger for chat sessions updated_at
CREATE TRIGGER update_chat_sessions_updated_at
    BEFORE UPDATE ON chat_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Enable RLS for chat sessions
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;

-- Users can view their own chat sessions
CREATE POLICY "Users can view own chat sessions" ON chat_sessions
    FOR SELECT USING (auth.uid()::text = user_id::text);

-- Users can create their own chat sessions
CREATE POLICY "Users can create chat sessions" ON chat_sessions
    FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

-- Users can update their own chat sessions (includes adding messages)
CREATE POLICY "Users can update own chat sessions" ON chat_sessions
    FOR UPDATE USING (auth.uid()::text = user_id::text);

-- Users can delete their own chat sessions
CREATE POLICY "Users can delete own chat sessions" ON chat_sessions
    FOR DELETE USING (auth.uid()::text = user_id::text);