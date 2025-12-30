#!/usr/bin/env python3
"""
Database seed script for ensureStudy
Creates default test accounts for all user types.

Usage:
    python seed_database.py
"""
import os
import sys

# Add the core-service to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend/core-service'))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def seed():
    """Seed the database with test accounts."""
    from app import create_app, db
    from app.models.user import User
    from app.models.organization import Organization
    from app.models.classroom import Classroom, StudentClassroom
    
    app = create_app()
    
    with app.app_context():
        print("🌱 Seeding database...")
        
        # Check if already seeded
        existing_user = User.query.filter_by(email="teacher@test.com").first()
        if existing_user:
            print("⚠️ Database already seeded. Skipping...")
            return True
        
        # Create Organization (School)
        print("📦 Creating organization...")
        org = Organization(
            name="Test School",
            email="school@testschool.com",
            phone="+91-9876543210",
            city="Mumbai",
            state="Maharashtra",
            country="India",
            license_count=100,
            used_licenses=0,
            admission_open=True,
            subscription_status="active"
        )
        db.session.add(org)
        db.session.flush()  # Get org.id
        
        # Create School Admin
        print("👤 Creating school admin...")
        admin = User(
            email="admin@test.com",
            username="admin",
            role="admin",
            first_name="Admin",
            last_name="User",
            organization_id=org.id,
            is_active=True,
            email_verified=True
        )
        admin.set_password("admin123")
        db.session.add(admin)
        db.session.flush()  # Get admin.id
        
        # Link admin to organization
        org.admin_user_id = admin.id
        
        # Create Teacher
        print("👤 Creating teacher...")
        teacher = User(
            email="teacher@test.com",
            username="teacher",
            role="teacher",
            first_name="Test",
            last_name="Teacher",
            organization_id=org.id,
            is_active=True,
            email_verified=True
        )
        teacher.set_password("teacher123")
        db.session.add(teacher)
        db.session.flush()  # Get teacher.id
        
        # Create Classroom
        print("🏫 Creating classroom...")
        classroom = Classroom(
            name="Physics Class 10-A",
            grade="10",
            section="A",
            subject="Physics",
            teacher_id=teacher.id,
            organization_id=org.id,
            is_active=True
        )
        db.session.add(classroom)
        db.session.flush()  # Get classroom.id
        
        # Create Students
        print("👤 Creating students...")
        students = []
        for i in range(1, 4):  # Create 3 students
            student = User(
                email=f"student{i}@test.com",
                username=f"student{i}",
                role="student",
                first_name=f"Student",
                last_name=f"{i}",
                organization_id=org.id,
                is_active=True,
                email_verified=True
            )
            student.set_password("student123")
            db.session.add(student)
            students.append(student)
        
        db.session.flush()  # Get student IDs
        
        # Enroll students in classroom
        print("📝 Enrolling students in classroom...")
        for student in students:
            enrollment = StudentClassroom(
                student_id=student.id,
                classroom_id=classroom.id,
                is_active=True
            )
            db.session.add(enrollment)
            org.used_licenses += 1
        
        # Create Parent
        print("👤 Creating parent...")
        parent = User(
            email="parent@test.com",
            username="parent",
            role="parent",
            first_name="Test",
            last_name="Parent",
            organization_id=org.id,
            is_active=True,
            email_verified=True
        )
        parent.set_password("parent123")
        db.session.add(parent)
        
        # Commit all changes
        db.session.commit()
        
        print("\n" + "="*50)
        print("✅ Database seeded successfully!")
        print("="*50)
        print("\n📋 TEST ACCOUNTS:")
        print("-"*50)
        print(f"{'Role':<12} {'Email':<25} {'Password':<15}")
        print("-"*50)
        print(f"{'Admin':<12} {'admin@test.com':<25} {'admin123':<15}")
        print(f"{'Teacher':<12} {'teacher@test.com':<25} {'teacher123':<15}")
        print(f"{'Student 1':<12} {'student1@test.com':<25} {'student123':<15}")
        print(f"{'Student 2':<12} {'student2@test.com':<25} {'student123':<15}")
        print(f"{'Student 3':<12} {'student3@test.com':<25} {'student123':<15}")
        print(f"{'Parent':<12} {'parent@test.com':<25} {'parent123':<15}")
        print("-"*50)
        print(f"\n🏫 ORGANIZATION:")
        print(f"   Name: Test School")
        print(f"   Access Token: {org.access_token}")
        print(f"   (Teachers use this token to register)")
        print(f"\n📚 CLASSROOM:")
        print(f"   Name: {classroom.name}")
        print(f"   Join Code: {classroom.join_code}")
        print(f"   (Students use this code to join)")
        print("="*60)
        
        return True


if __name__ == "__main__":
    seed()
