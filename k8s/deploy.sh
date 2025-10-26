#!/bin/bash

# AudioProcessor Kubernetes Deployment Script
# This script deploys the AudioProcessor to Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="ml-service"
MONITORING_NAMESPACE="monitoring"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-your-registry.com}"

echo -e "${BLUE}🚀 Starting AudioProcessor Kubernetes Deployment${NC}"
echo -e "${BLUE}================================================${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

print_status "Kubernetes cluster is accessible"

# Create namespaces
echo -e "${BLUE}📦 Creating namespaces...${NC}"
kubectl apply -f namespace.yaml
print_status "Namespaces created"

# Apply RBAC
echo -e "${BLUE}🔐 Setting up RBAC...${NC}"
kubectl apply -f rbac.yaml
print_status "RBAC configured"

# Apply ConfigMaps and Secrets
echo -e "${BLUE}⚙️  Applying configuration...${NC}"
kubectl apply -f configmap.yaml
print_warning "Please update secret.yaml with your actual secrets before applying"
read -p "Have you updated the secrets? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kubectl apply -f secret.yaml
    print_status "Secrets applied"
else
    print_warning "Skipping secrets - please apply manually later"
fi

# Apply services
echo -e "${BLUE}🌐 Creating services...${NC}"
kubectl apply -f services.yaml
print_status "Services created"

# Apply deployments
echo -e "${BLUE}🚀 Deploying AudioProcessor...${NC}"
kubectl apply -f audio-processor-deployment.yaml
kubectl apply -f audio-worker-deployment.yaml
print_status "AudioProcessor deployed"

# Apply monitoring
echo -e "${BLUE}📊 Setting up monitoring...${NC}"
kubectl apply -f monitoring.yaml
print_status "Monitoring configured"

# Apply HPA
echo -e "${BLUE}📈 Configuring autoscaling...${NC}"
kubectl apply -f hpa.yaml
print_status "Autoscaling configured"

# Apply Ingress
echo -e "${BLUE}🌍 Setting up ingress...${NC}"
kubectl apply -f ingress.yaml
print_status "Ingress configured"

# Wait for deployments to be ready
echo -e "${BLUE}⏳ Waiting for deployments to be ready...${NC}"
kubectl wait --for=condition=available --timeout=300s deployment/audio-processor -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/audio-worker -n $NAMESPACE
print_status "Deployments are ready"

# Check pod status
echo -e "${BLUE}🔍 Checking pod status...${NC}"
kubectl get pods -n $NAMESPACE
kubectl get pods -n $MONITORING_NAMESPACE

# Display service URLs
echo -e "${BLUE}🌐 Service URLs:${NC}"
echo -e "${GREEN}AudioProcessor API:${NC} http://ml-service.example.com/audio"
echo -e "${GREEN}Flower (Celery):${NC} http://ml-service.example.com/flower"
echo -e "${GREEN}Prometheus:${NC} http://monitoring.example.com/prometheus"
echo -e "${GREEN}Grafana:${NC} http://monitoring.example.com/grafana"
echo -e "${GREEN}AlertManager:${NC} http://monitoring.example.com/alertmanager"

# Display useful commands
echo -e "${BLUE}🛠️  Useful commands:${NC}"
echo -e "${YELLOW}# Check pod logs:${NC}"
echo "kubectl logs -f deployment/audio-processor -n $NAMESPACE"
echo "kubectl logs -f deployment/audio-worker -n $NAMESPACE"
echo ""
echo -e "${YELLOW}# Scale workers:${NC}"
echo "kubectl scale deployment audio-worker --replicas=10 -n $NAMESPACE"
echo ""
echo -e "${YELLOW}# Check HPA status:${NC}"
echo "kubectl get hpa -n $NAMESPACE"
echo ""
echo -e "${YELLOW}# Port forward for local testing:${NC}"
echo "kubectl port-forward service/audio-processor-service 8000:8000 -n $NAMESPACE"

print_status "AudioProcessor deployment completed successfully!"
echo -e "${BLUE}================================================${NC}"
