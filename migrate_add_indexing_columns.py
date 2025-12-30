#!/usr/bin/env python3
"""
Database migration script to add indexing columns to classroom_materials table.
Uses PostgreSQL via DATABASE_URL from .env

Usage:
    python migrate_add_indexing_columns.py
"""
import os
import sys

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed, using environment variables only")

def migrate():
    """Add indexing columns to classroom_materials table in PostgreSQL."""
    
    # Get PostgreSQL connection URL
    database_url = os.environ.get('DATABASE_URL')
    
    if not database_url:
        print("❌ DATABASE_URL not set. Please set it in .env or environment.")
        print("   Example: postgresql://user:pass@localhost:5432/dbname")
        return False
    
    if 'sqlite' in database_url.lower():
        print("❌ SQLite detected in DATABASE_URL. This script is for PostgreSQL only.")
        print(f"   Current: {database_url}")
        return False
    
    print(f"📦 Connecting to PostgreSQL...")
    
    try:
        import psycopg2
        from psycopg2 import sql
    except ImportError:
        print("❌ psycopg2 not installed. Run: pip install psycopg2-binary")
        return False
    
    try:
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        cursor = conn.cursor()
        print("✅ Connected to PostgreSQL")
        
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'classroom_materials'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print("⚠️ Table 'classroom_materials' does not exist yet.")
            print("   It will be created when the Flask app starts with db.create_all()")
            conn.close()
            return True
        
        # Check existing columns
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'classroom_materials';
        """)
        existing_columns = {row[0] for row in cursor.fetchall()}
        print(f"ℹ️ Existing columns: {', '.join(sorted(existing_columns))}")
        
        # Columns to add
        new_columns = [
            ("indexing_status", "VARCHAR(20) DEFAULT 'pending'"),
            ("indexed_at", "TIMESTAMP"),
            ("chunk_count", "INTEGER DEFAULT 0"),
            ("indexing_error", "TEXT")
        ]
        
        added = 0
        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                try:
                    cursor.execute(
                        sql.SQL("ALTER TABLE classroom_materials ADD COLUMN {} {}").format(
                            sql.Identifier(col_name),
                            sql.SQL(col_type)
                        )
                    )
                    print(f"✅ Added column: {col_name}")
                    added += 1
                except psycopg2.Error as e:
                    print(f"⚠️ Error adding {col_name}: {e}")
            else:
                print(f"ℹ️ Column already exists: {col_name}")
        
        conn.close()
        
        if added > 0:
            print(f"\n✅ Migration complete! Added {added} columns.")
        else:
            print("\n✅ All columns already exist. No changes needed.")
        
        return True
        
    except psycopg2.Error as e:
        print(f"❌ PostgreSQL connection error: {e}")
        print("\nMake sure PostgreSQL is running:")
        print("   docker-compose up -d postgres")
        print("   or: brew services start postgresql")
        return False

if __name__ == "__main__":
    success = migrate()
    sys.exit(0 if success else 1)
