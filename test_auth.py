#!/usr/bin/env python3
"""
Test script for authentication functionality
"""

import os
from dotenv import load_dotenv
from ai_modules.auth import AuthManager

def test_auth_setup():
    """Test if authentication is properly configured"""
    print("Testing Authentication Setup...")
    
    # Load environment variables
    load_dotenv()
    
    # Check if environment variables are set
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    
    if not url or not key:
        print("❌ Supabase credentials not found!")
        print("Please create a .env file with your Supabase credentials:")
        print("SUPABASE_URL=your_supabase_project_url")
        print("SUPABASE_ANON_KEY=your_supabase_anon_key")
        return False
    
    print("✅ Environment variables found")
    print(f"URL: {url[:30]}...")
    print(f"Key: {key[:20]}...")
    
    # Test Supabase client initialization
    try:
        auth_manager = AuthManager()
        if auth_manager.client:
            print("✅ Supabase client initialized successfully")
            return True
        else:
            print("❌ Failed to initialize Supabase client")
            return False
    except Exception as e:
        print(f"❌ Error initializing Supabase client: {str(e)}")
        return False

if __name__ == "__main__":
    test_auth_setup()
