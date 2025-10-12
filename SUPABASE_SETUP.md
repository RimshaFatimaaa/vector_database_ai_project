# Supabase Authentication Setup Guide

This guide will help you set up Supabase authentication for your AI Interview Coach app.

## 🔧 Required Supabase Settings

### 1. Create a Supabase Project

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up or log in to your account
3. Click "New Project"
4. Choose your organization
5. Enter project details:
   - **Name**: `ai-interview-coach` (or any name you prefer)
   - **Database Password**: Create a strong password
   - **Region**: Choose the closest region to your users
6. Click "Create new project"

### 2. Get Your Project Credentials

Once your project is created:

1. Go to **Settings** → **API** in your Supabase dashboard
2. Copy the following values:
   - **Project URL** (looks like: `https://your-project-id.supabase.co`)
   - **anon/public key** (starts with `eyJ...`)

### 3. Configure Environment Variables

1. Copy the `supabase_config.env.example` file to `.env`:
   ```bash
   cp supabase_config.env.example .env
   ```

2. Edit the `.env` file and add your Supabase credentials:
   ```env
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

### 4. Configure Authentication Settings

In your Supabase dashboard:

1. Go to **Authentication** → **Settings**
2. Configure the following:

#### Site URL
- Set to: `http://localhost:8501` (for local development)
- For production: `https://your-domain.com`

#### Redirect URLs
- Add: `http://localhost:8501/**` (for local development)
- Add: `https://your-domain.com/**` (for production)

#### Email Settings
- **Enable email confirmations**: ✅ (recommended)
- **Enable email change confirmations**: ✅ (recommended)
- **Enable phone confirmations**: ❌ (optional)

#### Password Settings
- **Minimum password length**: 6
- **Require special characters**: ❌ (optional)
- **Require numbers**: ❌ (optional)
- **Require uppercase letters**: ❌ (optional)

### 5. User Management (Optional)

If you want to manage users manually:

1. Go to **Authentication** → **Users**
2. You can view, edit, or delete users here
3. You can also manually create users if needed

## 🚀 Running the App

1. Install the updated requirements:
   ```bash
   pip install -r demo_requirements.txt
   ```

2. Make sure your `.env` file is in the project root directory

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser to `http://localhost:8501`

## 🔐 Authentication Flow

1. **First Visit**: Users see the login/signup page
2. **Sign Up**: New users can create accounts (email verification required)
3. **Login**: Existing users can sign in
4. **Protected Access**: Only authenticated users can access the NLP demo
5. **Logout**: Users can sign out from the sidebar

## 🛠️ Troubleshooting

### Common Issues:

1. **"Supabase credentials not found"**
   - Make sure your `.env` file exists and contains the correct credentials
   - Check that the file is in the project root directory

2. **"Invalid email or password"**
   - Verify the user exists in Supabase dashboard
   - Check if email verification is required

3. **"Sign up failed"**
   - Check Supabase logs in the dashboard
   - Verify email settings are configured correctly

4. **App not loading**
   - Check that all dependencies are installed
   - Verify the `.env` file format is correct

### Getting Help:

- Check Supabase documentation: [https://supabase.com/docs](https://supabase.com/docs)
- View your project logs in the Supabase dashboard
- Check the Streamlit terminal for error messages

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Use environment variables in production
- Regularly rotate your API keys
- Monitor user activity in the Supabase dashboard
