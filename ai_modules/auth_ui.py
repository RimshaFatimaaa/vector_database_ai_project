"""
Authentication UI components for Streamlit
Handles login and signup forms
"""

import streamlit as st
from ai_modules.auth import AuthManager

def show_login_form():
    """Display login form"""
    st.markdown("### 🔐 Login to Your Account")
    
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="Enter your email address")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            login_clicked = st.form_submit_button("Login", type="primary", use_container_width=True)
        with col2:
            switch_to_signup = st.form_submit_button("Switch to Sign Up", use_container_width=True)
        
        if login_clicked:
            if email and password:
                auth_manager = AuthManager()
                result = auth_manager.sign_in(email, password)
                
                if result.get("success"):
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.session_state.user_name = email.split('@')[0]
                    st.rerun()
                else:
                    st.error(result.get("error", "Login failed"))
            else:
                st.error("Please fill in all fields")
        
        if switch_to_signup:
            st.session_state.show_signup = True
            st.rerun()

def show_signup_form():
    """Display signup form"""
    st.markdown("### 📝 Create New Account")
    
    with st.form("signup_form"):
        full_name = st.text_input("Full Name", placeholder="Enter your full name")
        email = st.text_input("Email", placeholder="Enter your email address")
        password = st.text_input("Password", type="password", placeholder="Create a password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            signup_clicked = st.form_submit_button("Sign Up", type="primary", use_container_width=True)
        with col2:
            switch_to_login = st.form_submit_button("Switch to Login", use_container_width=True)
        
        if signup_clicked:
            if email and password and confirm_password:
                if password == confirm_password:
                    if len(password) >= 6:
                        auth_manager = AuthManager()
                        result = auth_manager.sign_up(email, password, full_name)
                        
                        if result.get("success"):
                            st.info("Please check your email to verify your account before logging in.")
                            st.session_state.show_signup = False
                            st.rerun()
                        else:
                            st.error(result.get("error", "Sign up failed"))
                    else:
                        st.error("Password must be at least 6 characters long")
                else:
                    st.error("Passwords do not match")
            else:
                st.error("Please fill in all fields")
        
        if switch_to_login:
            st.session_state.show_signup = False
            st.rerun()

def show_auth_page():
    """Display the main authentication page"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #1f77b4; margin-bottom: 0.5rem;">🎤 AI Interview Coach</h1>
        <p style="color: #666; font-size: 1.1rem;">Please sign in or create an account to access the NLP Analysis Demo</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    
    # Create two columns for login/signup forms
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        if not st.session_state.show_signup:
            show_login_form()
        else:
            show_signup_form()
    
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px; border-left: 4px solid #1f77b4;">
            <h3 style="color: #1f77b4; margin-top: 0;">🔒 Secure Authentication</h3>
            <p style="color: #666; margin-bottom: 0;">
                Your data is protected with enterprise-grade security. 
                We use Supabase for secure authentication and data management.
            </p>
        </div>
        """, unsafe_allow_html=True)

def show_logout_button():
    """Display logout button in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Welcome, {st.session_state.user_name}!**")
        
        if st.button("🚪 Logout", use_container_width=True):
            auth_manager = AuthManager()
            result = auth_manager.sign_out()
            
            if result.get("success"):
                st.session_state.authenticated = False
                st.session_state.user_email = None
                st.session_state.user_name = None
                st.rerun()
            else:
                st.error("Logout failed. Please try again.")

def show_header_logout():
    """Display logout button in the main page header"""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">🎤 AI Interview Coach - NLP Analysis Demo</h1>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        if st.button("🚪 Logout", key="header_logout", use_container_width=True, type="secondary"):
            auth_manager = AuthManager()
            result = auth_manager.sign_out()
            
            if result.get("success"):
                st.session_state.authenticated = False
                st.session_state.user_email = None
                st.session_state.user_name = None
                st.rerun()
            else:
                st.error("Logout failed. Please try again.")
