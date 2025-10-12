"""
Authentication module using Supabase
Handles user login, signup, and session management
"""

import streamlit as st
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class AuthManager:
    def __init__(self):
        """Initialize Supabase client"""
        try:
            # Use st.secrets directly for Streamlit Cloud
            self.url = st.secrets["SUPABASE_URL"]
            self.key = st.secrets["SUPABASE_ANON_KEY"]
            self.client: Client = create_client(self.url, self.key)
        except KeyError as e:
            st.error(f"⚠️ Missing Supabase credentials: {str(e)}. Please set SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit Cloud secrets.")
            self.client = None
        except Exception as e:
            st.error(f"⚠️ Failed to initialize Supabase client: {str(e)}")
            self.client = None
    
    def sign_up(self, email: str, password: str, full_name: str = None) -> dict:
        """Sign up a new user"""
        if not self.client:
            return {"error": "Supabase client not initialized"}
        
        try:
            # Create user with email and password
            response = self.client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "full_name": full_name or email.split('@')[0]
                    }
                }
            })
            
            if response.user:
                return {
                    "success": True,
                    "message": "Account created successfully! Please check your email to verify your account.",
                    "user": response.user
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create account. Please try again."
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Sign up failed: {str(e)}"
            }
    
    def sign_in(self, email: str, password: str) -> dict:
        """Sign in an existing user"""
        if not self.client:
            return {"error": "Supabase client not initialized"}
        
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.user:
                return {
                    "success": True,
                    "message": "Login successful!",
                    "user": response.user
                }
            else:
                return {
                    "success": False,
                    "error": "Invalid email or password"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Login failed: {str(e)}"
            }
    
    def sign_out(self) -> dict:
        """Sign out the current user"""
        if not self.client:
            return {"error": "Supabase client not initialized"}
        
        try:
            self.client.auth.sign_out()
            return {
                "success": True,
                "message": "Logged out successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Logout failed: {str(e)}"
            }
    
    def get_current_user(self):
        """Get the current authenticated user"""
        if not self.client:
            return None
        
        try:
            user = self.client.auth.get_user()
            return user.user if user else None
        except:
            return None
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        user = self.get_current_user()
        return user is not None

def init_session_state():
    """Initialize session state for authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None

def check_auth_status():
    """Check and update authentication status"""
    # First check if we already have session state authentication
    if st.session_state.get('authenticated', False):
        return True
    
    # If not, check with Supabase
    auth_manager = AuthManager()
    is_auth = auth_manager.is_authenticated()
    
    if is_auth:
        user = auth_manager.get_current_user()
        if user:
            st.session_state.authenticated = True
            st.session_state.user_email = user.email
            st.session_state.user_name = user.user_metadata.get('full_name', user.email.split('@')[0])
    else:
        st.session_state.authenticated = False
        st.session_state.user_email = None
        st.session_state.user_name = None
    
    return st.session_state.authenticated
