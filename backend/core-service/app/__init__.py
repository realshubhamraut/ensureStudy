"""
Flask Core Service Application Factory
"""
import os
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()


def create_app(config_name=None):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('JWT_SECRET', 'dev-secret-key')
    
    # INTEGRATION NOTE: Always use PostgreSQL from DATABASE_URL
    # Removed SQLite fallback per user request - consistent DB across all environments
    database_url = os.getenv(
        'DATABASE_URL', 
        'postgresql://ensure_study_user:secure_password_123@localhost:5432/ensure_study'
    )
    print(f"[Database] Using PostgreSQL: {database_url.split('@')[1] if '@' in database_url else 'configured'}")
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # File upload settings
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.users import users_bp
    from app.routes.progress import progress_bp
    from app.routes.assessments import assessments_bp
    from app.routes.leaderboard import leaderboard_bp
    from app.routes.admin import admin_bp
    from app.routes.curriculum import curriculum_bp
    from app.routes.teacher import teacher_bp
    from app.routes.classroom import classroom_bp
    from app.routes.students import bp as students_bp
    from app.routes.notes import notes_bp
    from app.routes.assignment import assignment_bp
    from app.routes.files import files_bp
    from app.routes.evaluation import evaluation_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(users_bp)
    app.register_blueprint(progress_bp)
    app.register_blueprint(assessments_bp)
    app.register_blueprint(leaderboard_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(curriculum_bp)
    app.register_blueprint(teacher_bp)
    app.register_blueprint(classroom_bp)
    app.register_blueprint(students_bp)
    app.register_blueprint(notes_bp)
    app.register_blueprint(assignment_bp)
    app.register_blueprint(files_bp)
    app.register_blueprint(evaluation_bp)
    
    # Import models for table creation
    from app.models.organization import Organization, LicensePurchase
    from app.models.student_profile import StudentProfile, ParentStudentLink, TeacherClassAssignment
    from app.models.curriculum import Subject, Topic, Subtopic, SubtopicAssessment, StudentSubtopicProgress
    from app.models.classroom import Classroom, StudentClassroom
    from app.models.notes import NoteProcessingJob, DigitizedNotePage, NoteEmbedding, NoteSearchHistory
    from app.models.assignment import Assignment, AssignmentAttachment, Submission, SubmissionFile
    from app.models.exam_evaluation import ExamSession, StudentEvaluation
    
    # Health check endpoint
    @app.route('/health')
    def health():
        return {'status': 'healthy', 'service': 'core-api'}
    
    # Create tables
    with app.app_context():
        db.create_all()
    
    return app
