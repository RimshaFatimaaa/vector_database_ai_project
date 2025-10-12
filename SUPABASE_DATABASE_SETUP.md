# Supabase Database Setup Guide

## 🔧 **Step 1: Run SQL Setup Script**

1. Go to your Supabase dashboard
2. Navigate to **SQL Editor**
3. Copy and paste the contents of `supabase_setup.sql`
4. Click **Run** to execute the script

## ⚙️ **Step 2: Configure Authentication Settings**

### **Authentication → Settings:**

1. **Site URL**: `http://localhost:8501`
2. **Redirect URLs**: Add `http://localhost:8501/**`
3. **JWT expiry**: `3600` (1 hour)
4. **Refresh token expiry**: `2592000` (30 days)

### **Email Settings:**
- ✅ **Enable email confirmations**: ON
- ✅ **Enable email change confirmations**: ON
- ❌ **Enable phone confirmations**: OFF (optional)

### **Password Settings:**
- **Minimum password length**: `6`
- ❌ **Require special characters**: OFF
- ❌ **Require numbers**: OFF
- ❌ **Require uppercase letters**: OFF

## 🔐 **Step 3: Configure Email Templates (Optional)**

### **Email Confirmation Template:**
```
Subject: Confirm your email for AI Interview Coach

Hi there,

Please confirm your email address by clicking the link below:

{{ .ConfirmationURL }}

If you didn't create an account, you can safely ignore this email.

Best regards,
AI Interview Coach Team
```

### **Password Reset Template:**
```
Subject: Reset your password for AI Interview Coach

Hi there,

You can reset your password by clicking the link below:

{{ .ConfirmationURL }}

If you didn't request this, you can safely ignore this email.

Best regards,
AI Interview Coach Team
```

## 🧪 **Step 4: Test Authentication**

1. **Run the app**: `streamlit run app.py`
2. **Open**: `http://localhost:8501`
3. **Try signing up** with a test email
4. **Check your email** for verification
5. **Click the verification link**
6. **Try logging in**

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **"Invalid login credentials"**
   - Check if email is verified
   - Verify password is correct
   - Check Supabase logs

2. **"Email not confirmed"**
   - Check spam folder
   - Verify email template is configured
   - Check Supabase logs

3. **"User not found"**
   - Check if user exists in auth.users table
   - Verify profiles table was created

4. **"Permission denied"**
   - Check RLS policies
   - Verify user has correct permissions

### **Check Database Tables:**

1. Go to **Table Editor** in Supabase
2. Verify these tables exist:
   - `auth.users` (auto-created)
   - `public.profiles` (created by our script)

### **Check Logs:**

1. Go to **Logs** in Supabase dashboard
2. Look for authentication errors
3. Check for any permission issues

## 📊 **Database Schema:**

```sql
-- auth.users (auto-created by Supabase)
- id (UUID, Primary Key)
- email (TEXT)
- encrypted_password (TEXT)
- email_confirmed_at (TIMESTAMP)
- created_at (TIMESTAMP)
- updated_at (TIMESTAMP)
- raw_user_meta_data (JSONB)

-- public.profiles (created by our script)
- id (UUID, Foreign Key to auth.users)
- full_name (TEXT)
- email (TEXT)
- created_at (TIMESTAMP)
- updated_at (TIMESTAMP)
```

## 🚀 **After Setup:**

Once you've completed these steps, your authentication should work properly. Users will be able to:
- Sign up with email and password
- Receive email verification
- Login after verification
- Access the NLP demo features
- Logout securely
