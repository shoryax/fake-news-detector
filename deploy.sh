#!/bin/bash

# Deployment script for Azure VM
echo "Starting deployment..."

# Variables (UPDATE THESE)
USERNAME="YOUR_USERNAME"
VM_IP="YOUR_VM_IP"
PROJECT_DIR="/home/$USERNAME/news"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}1. Transferring files to VM...${NC}"
rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude '*.pyc' \
    /Users/shoryavardhan/projects/news/ $USERNAME@$VM_IP:$PROJECT_DIR/

echo -e "${GREEN}2. Installing dependencies...${NC}"
ssh $USERNAME@$VM_IP << 'ENDSSH'
cd ~/news
source venv/bin/activate
pip install -r requirements.txt

# Download NLTK data if needed
python3 << EOF
import nltk
nltk.download('stopwords')
EOF

ENDSSH

echo -e "${GREEN}3. Restarting service...${NC}"
ssh $USERNAME@$VM_IP "sudo systemctl restart news-app"

echo -e "${GREEN}Deployment complete!${NC}"
echo "Visit http://$VM_IP to see your app"
