"""
Organization Model for Multi-Tenant School System
"""
from datetime import datetime
import uuid
import secrets
from app import db


def generate_uuid():
    return str(uuid.uuid4())


def generate_access_token():
    """Generate a unique access token for organization"""
    return secrets.token_urlsafe(16)


class Organization(db.Model):
    """School/Organization that purchases licenses"""
    __tablename__ = "organizations"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    city = db.Column(db.String(100))
    state = db.Column(db.String(100))
    country = db.Column(db.String(100), default="India")
    
    # Licensing
    license_count = db.Column(db.Integer, default=0)  # Total purchased
    used_licenses = db.Column(db.Integer, default=0)  # Currently used
    price_per_student = db.Column(db.Float, default=29.0)  # INR
    
    # Access token for registration
    access_token = db.Column(db.String(32), unique=True, default=generate_access_token)
    
    # Subscription status
    subscription_status = db.Column(db.String(20), default="trial")  # trial, active, expired
    subscription_expires = db.Column(db.DateTime)
    
    # Admission window control
    admission_open = db.Column(db.Boolean, default=True)  # When False, tokens can't be used
    
    # Admin user
    admin_user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    users = db.relationship("User", foreign_keys="User.organization_id", backref="organization", lazy="dynamic")
    
    def has_available_licenses(self) -> bool:
        """Check if organization has available student licenses"""
        return self.used_licenses < self.license_count
    
    def use_license(self):
        """Use one license when a student registers"""
        if self.has_available_licenses():
            self.used_licenses += 1
            return True
        return False
    
    def release_license(self):
        """Release a license when a student is removed"""
        if self.used_licenses > 0:
            self.used_licenses -= 1
    
    def regenerate_token(self):
        """Regenerate access token"""
        self.access_token = generate_access_token()
        return self.access_token
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "license_count": self.license_count,
            "used_licenses": self.used_licenses,
            "available_licenses": self.license_count - self.used_licenses,
            "access_token": self.access_token,
            "subscription_status": self.subscription_status,
            "subscription_expires": self.subscription_expires.isoformat() if self.subscription_expires else None,
            "admission_open": self.admission_open,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class LicensePurchase(db.Model):
    """Record of license purchases"""
    __tablename__ = "license_purchases"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    organization_id = db.Column(db.String(36), db.ForeignKey("organizations.id"), nullable=False)
    
    # Purchase details
    quantity = db.Column(db.Integer, nullable=False)
    price_per_unit = db.Column(db.Float, nullable=False)
    total_amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(3), default="INR")
    
    # Payment details (Razorpay)
    payment_id = db.Column(db.String(100))
    payment_status = db.Column(db.String(20), default="pending")  # pending, completed, failed
    payment_method = db.Column(db.String(50))
    
    # Invoice
    invoice_number = db.Column(db.String(50))
    invoice_url = db.Column(db.String(500))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    organization = db.relationship("Organization", backref="purchases")
    
    def to_dict(self):
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "quantity": self.quantity,
            "price_per_unit": self.price_per_unit,
            "total_amount": self.total_amount,
            "currency": self.currency,
            "payment_status": self.payment_status,
            "invoice_number": self.invoice_number,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
