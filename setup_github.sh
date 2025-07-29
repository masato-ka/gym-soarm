#!/bin/bash

# GitHub Repository Setup Script for gym-soarm
# Run this script after creating the repository on GitHub.com

set -e

echo "🚀 Setting up GitHub repository for gym-soarm..."

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "gym_soarm" ]; then
    echo "❌ Error: Please run this script from the gym-soarm project root directory"
    exit 1
fi

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Error: Git repository not initialized. Run 'git init' first."
    exit 1
fi

# Prompt for GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ Error: GitHub username is required"
    exit 1
fi

# Set up the remote
REPO_URL="https://github.com/${GITHUB_USERNAME}/gym-soarm.git"
echo "📡 Adding remote origin: $REPO_URL"

# Remove existing origin if it exists
git remote remove origin 2>/dev/null || true

# Add new origin
git remote add origin "$REPO_URL"

# Verify remote was added
echo "✅ Remote added successfully:"
git remote -v

# Prepare main branch and push
echo "📤 Pushing to GitHub..."
git branch -M main

# Attempt to push
if git push -u origin main; then
    echo "🎉 Successfully pushed to GitHub!"
    echo "📍 Repository URL: https://github.com/${GITHUB_USERNAME}/gym-soarm"
    echo ""
    echo "Next steps:"
    echo "1. Visit your repository to verify all files are uploaded"
    echo "2. Add repository topics: robotics, reinforcement-learning, gymnasium, mujoco, so-arm"
    echo "3. Consider enabling Issues and Discussions for community engagement"
else
    echo "❌ Failed to push to GitHub. Please check:"
    echo "1. Repository exists on GitHub: https://github.com/${GITHUB_USERNAME}/gym-soarm"
    echo "2. Repository is empty (no README, .gitignore, or license initialized)"
    echo "3. You have push access to the repository"
    echo "4. Your Git credentials are configured correctly"
    echo ""
    echo "Manual setup command:"
    echo "git push -u origin main"
fi