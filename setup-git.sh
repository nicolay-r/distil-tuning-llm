#!/bin/bash

# Set your Git user details and repository SSH URL
GIT_USERNAME="Your GitHub Username"
GIT_EMAIL="your.email@example.com"
GITHUB_TOKEN="YOUR_GITHUB_TOKEN"
REPO_SSH_URL="git@github.com:USERNAME/REPOSITORY.git"  # Replace with your repository's SSH URL


# Check if SSH key already exists, generate one if it doesn't
SSH_KEY_PATH="$HOME/.ssh/id_rsa"
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "SSH key not found, generating one..."
    ssh-keygen -t rsa -b 4096 -C "$GIT_EMAIL" -N "" -f "$SSH_KEY_PATH"
    echo "SSH key generated."
else
    echo "SSH key already exists."
fi

# Start the ssh-agent in the background
eval "$(ssh-agent -s)"

# Add your SSH key to the ssh-agent
ssh-add "$SSH_KEY_PATH"

# Configure Git with your name and email
git config --global user.name "$GIT_USERNAME"
git config --global user.email "$GIT_EMAIL"

# Set Git to use the credential memory cache
git config --global credential.helper cache
# Set the cache to timeout after 1 hour (change as needed)
git config --global credential.helper 'cache --timeout=3600'

# Change the Git remote URL to SSH
git remote set-url origin "$REPO_SSH_URL"

# Upload the SSH key to your GitHub account
KEY_TITLE="Key for $(hostname)"
KEY_CONTENT="$(cat "$SSH_KEY_PATH.pub")"

# Use the GitHub API to add the SSH key
curl -X POST -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/user/keys \
    -d "{\"title\": \"$KEY_TITLE\", \"key\": \"$KEY_CONTENT\"}"

echo "Git configuration is complete!"